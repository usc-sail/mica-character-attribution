"""Instruction Fine-tune LLMs on the character attribution classification task"""
import datadirs
import data_utils

from absl import app
from absl import flags
from absl import logging
from accelerate import PartialState
from datasets import Dataset
from datetime import datetime
import json
import jsonlines
import numpy as np
import os
import pandas as pd
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers import DataCollatorWithPadding
from transformers.trainer import EvalPrediction
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from typing import Dict, List, Union, Tuple

flags.DEFINE_string("contexts_file", default=None, help="contexts data file")
flags.DEFINE_string("extracts_file", default=None, help="extracts data file")
flags.DEFINE_string("model", default="meta-llama/Llama-3.1-8B-Instruct", help="huggingface model name")
flags.DEFINE_bool("instrtune", default=False, help="do instruction tuning instead of classification")
flags.DEFINE_bool("bf16", default=False, help="use brain floating point (default=fp16)")
flags.DEFINE_bool("load_4bit", default=False, help="do 4-bit QLoRA training")
flags.DEFINE_bool("load_8bit", default=False, help="do 8-bit QLoRA training")
flags.DEFINE_enum("attn",
                  default="sdpa",
                  enum_values=["flash_attention_2", "sdpa", "eager"],
                  help="attention implementation")
flags.DEFINE_integer("dataset_batch_size", default=4096, help="dataset batch size for tokenization")
flags.DEFINE_integer("train_batch_size", default=1, help="training batch size")
flags.DEFINE_integer("eval_batch_size", default=1, help="evaluation batch size")
flags.DEFINE_integer("eval_accumulation_steps",
                     default=None,
                     help="number of prediction steps to accumulate the output tensors for, before moving to CPU")
flags.DEFINE_bool("eval_on_start", default=False, help="evaluate before training begins")
flags.DEFINE_float("lr", default=2e-5, help="learning rate")
flags.DEFINE_string("optim",
                    default="adamw_torch",
                    help=("optimizer name (https://github.com/huggingface/transformers/blob/main/src/transformers/"
                          "training_args.py)"))
flags.DEFINE_integer("max_seq_len", default=1024, help="maximum sequence length")
flags.DEFINE_integer("train_steps", default=1024, help="training steps")
flags.DEFINE_integer("eval_steps", default=32, help="training steps between successive evaluations")
flags.DEFINE_integer("logging_steps", default=8, help="training steps between successive logging")
flags.DEFINE_multi_string("lora_target_module", default=["q_proj", "k_proj"], help="target modules to train using LoRA")
flags.DEFINE_integer("rank", default=32, help="lora rank")
flags.DEFINE_integer("alpha", default=64, help="lora alpha")
flags.DEFINE_float("dropout", default=0, help="dropout")
flags.DEFINE_float("weight_decay", default=0, help="weight decay")
flags.DEFINE_float("max_grad_norm", default=10, help="maximum gradient norm")
flags.DEFINE_enum("metric",
                  default="f1",
                  enum_values=["accuracy", "f1", "precision", "recall"],
                  help="metric to compare models")
flags.DEFINE_bool("logtofile", default=False, help="log to file")
flags.DEFINE_bool("save_predictions", default=False,
                  help="save predictions of eval set corresponding to the best metric value")
flags.DEFINE_bool("save_model", default=False, help="save best model")

def input_checker(args):
    """Check at least one of contexts_file or extracts_file is given"""
    given_contexts = args["contexts_file"] is not None
    given_extracts = args["extracts_file"] is not None
    return (given_contexts and not given_extracts) or (not given_contexts and given_extracts)

flags.register_multi_flags_validator(flag_names=["contexts_file", "extracts_file"],
                                     multi_flags_checker=input_checker,
                                     message="Provide exactly one of contexts_file or extracts_file")
FLAGS = flags.FLAGS
PARTIALSTATE = PartialState()
TEMPLATE = ("Given the definition of a character attribute or trope, the name of a character, and a story or segments "
            "of a story where the character appears, speaks or is mentioned, answer 'yes' or 'no' if the character "
            "portrays or is associated with the attribute or trope in the story.\n\nATTRIBUTE: $ATTRIBUTE$"
            "\n\nCHARACTER: $CHARACTER$\n\nSTORY: $STORY$. \n\n ANSWER: $ANSWER$")
ANSWER_TEMPLATE = " \n\n ANSWER:"
CHARACTER_TOKEN = "[CHARACTER]"
ATTRIBUTE_TOKEN = "[ATTRIBUTE]"
CONTEXT_TOKEN = "[CONTEXT]"

random.seed(130194)

def log(message):
    if PARTIALSTATE.is_local_main_process:
        logging.info(message)

class LoggingCallback(TrainerCallback):
    """Callback class to log to file"""
    def __init__(self, logs_file=None):
        super().__init__()
        self.logs_writer = None
        if logs_file is not None:
            self.logs_writer = jsonlines.open(logs_file, mode="w", flush=True)

    def on_log(self,
               args: TrainingArguments,
               state: TrainerState,
               control: TrainerControl,
               logs: Dict[str, float],
               **kwargs):
        if state.is_local_process_zero:
            logging.info(f"STEP {state.global_step}/{state.max_steps}")
            for logkey, logvalue in logs.items():
                logging.info(f"{logkey} = {logvalue:.6f}")
            logging.info("\n")
            if self.logs_writer is not None:
                logs["step"] = state.global_step
                self.logs_writer.write(logs)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.logs_writer is not None:
            self.logs_writer.close()

def create_dataset(tokenizer: AutoTokenizer, data: List[Dict[str, Union[str, int]]]) -> Tuple[Dataset, pd.DataFrame]:
    """Create Dataset object and evaluation dataframe from data array"""
    texts = []
    rows = []
    for obj in data:
        if FLAGS.instrtune:
            text = (TEMPLATE
                    .replace("$ANSWER$", "yes" if obj["label"] == 1 else "no")
                    .replace("$CHARACTER$", obj["character"])
                    .replace("$ATTRIBUTE$", obj["attribute-definition"])
                    .replace("$STORY$", obj["text"])
                    )
        else:
            text = (f"{ATTRIBUTE_TOKEN}{obj['attribute-definition']}{CHARACTER_TOKEN}{obj['character']}{CONTEXT_TOKEN}"
                    f"{obj['text']}")
        row = [obj["key"], obj["attribute-name"], obj["label"]]
        rows.append(row)
        texts.append(text)
    df = pd.DataFrame(rows, columns=["key", "attribute", "label"])
    input_ids, attention_mask = [], []
    n_batches = int(np.ceil(len(texts)/FLAGS.dataset_batch_size))
    for i in tqdm.trange(n_batches, desc="tokenization", disable=not PARTIALSTATE.is_local_main_process):
        batch_texts = texts[FLAGS.dataset_batch_size * i: FLAGS.dataset_batch_size * (i + 1)]
        encoding = tokenizer(batch_texts,
                             padding=False,
                             add_special_tokens=True,
                             return_overflowing_tokens=False,
                             return_length=False)
        input_ids.extend(encoding["input_ids"])
        attention_mask.extend(encoding["attention_mask"])
    if FLAGS.instrtune:
        yes_answer_sp_mask = tokenizer(ANSWER_TEMPLATE + " yes",
                                       add_special_tokens=True,
                                       return_special_tokens_mask=True)["special_tokens_mask"]
        no_answer_sp_mask = tokenizer(ANSWER_TEMPLATE + " no",
                                       add_special_tokens=True,
                                       return_special_tokens_mask=True)["special_tokens_mask"]
        is_first_token_sp = yes_answer_sp_mask[0] == 1
        is_last_token_sp = yes_answer_sp_mask[-1] == 1
        yes_answer_size = len(yes_answer_sp_mask) - is_first_token_sp - is_last_token_sp
        no_answer_size = len(no_answer_sp_mask) - is_first_token_sp - is_last_token_sp
        for i in tqdm.trange(len(input_ids), desc="truncation", disable=not PARTIALSTATE.is_local_main_process):
            if len(input_ids[i]) > FLAGS.max_seq_len:
                if data[i]["label"] == 1:
                    start = -is_last_token_sp - yes_answer_size - (len(input_ids[i]) - FLAGS.max_seq_len)
                    end = -is_last_token_sp - yes_answer_size
                else:
                    start = -is_last_token_sp - no_answer_size - (len(input_ids[i]) - FLAGS.max_seq_len)
                    end = -is_last_token_sp - no_answer_size
                input_ids[i] = input_ids[i][:start] + input_ids[i][end:]
                attention_mask[i] = attention_mask[i][:start] + attention_mask[i][end:]
        dataset = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})
    else:
        dataset = Dataset.from_dict({"input_ids": input_ids,
                                     "attention_mask": attention_mask,
                                     "labels": df["label"].tolist()})
    return dataset, df

def preprocess_logits_for_metrics(logits: torch.Tensor, _) -> torch.Tensor:
    """Preprocess logits to prepare for metrics computation for the instruction-tuning method.
    We use this functions to save space primarily, converting batch-size x seqlen x vocab-size tensors to batch-size
    x seqlen x 2 tensors
    """
    probs = logits.softmax(dim=-1) # batch-size x seqlen x vocab-size
    max_output = torch.max(probs, dim=-1)
    z = torch.cat((max_output.values.unsqueeze(dim=-1),
                   max_output.indices.unsqueeze(dim=-1)), dim=-1) # batch-size x seqlen x 2
    return z

class ComputeMetrics:
    """Compute metrics class for classification and instruction-tuning methods for CHATTER, PERSONET & 
    STORY2PERSONALITY datasets"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eval_df = None
        self._dataset = "chatter"

    def set_dataset(self, dataset):
        if dataset not in ["chatter", "personet", "personality"]:
            raise ValueError("dataset should be either chatter, personet, or personality")
        self._dataset = dataset

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute metrics for different datasets"""
        if self._dataset == "chatter":
            true_arr, pred_arr, prob_arr = [], [], []
            for _, sample_df in self.eval_df.groupby("key"):
                nonnan_sample_df = sample_df.dropna(subset="pred")
                if len(nonnan_sample_df) > 0:
                    true = nonnan_sample_df["label"].values[0]
                    if any(nonnan_sample_df["pred"] == 1):
                        pred = 1
                        prob = max(nonnan_sample_df[nonnan_sample_df["pred"] == 1]["prob"].tolist())
                    else:
                        pred = 0
                        prob = max(nonnan_sample_df[nonnan_sample_df["pred"] == 0]["prob"].tolist())
                    true_arr.append(true)
                    pred_arr.append(pred)
                    prob_arr.append(prob)
            if len(true_arr) > 0:
                acc = accuracy_score(true_arr, pred_arr)
                auc = roc_auc_score(true_arr, prob_arr)
                prec, rec, f1, _ = precision_recall_fscore_support(true_arr, pred_arr, average="binary")
            else:
                acc, auc, prec, rec, f1 = np.nan, np.nan, np.nan, np.nan, np.nan
            metrics = {"accuracy": acc,
                       "precision": prec,
                       "recall": rec,
                       "f1": f1,
                       "auc": auc,
                       "total_nsamples": len(self.eval_df["key"].unique()),
                       "eval_nsamples": len(true_arr)}
        elif self._dataset == "personet":
            n1, n2, n3 = 0, 0, 0
            for _, sample_df in self.eval_df.groupby("key"):
                pos_sample_df = sample_df[sample_df["pred"] == 1]
                if len(pos_sample_df) > 0 and (pos_sample_df["label"] == 1).any():
                    i = pos_sample_df["label"].values.nonzero()[0].item()
                    rank = (-pos_sample_df["prob"].values).argsort()
                    if rank[i] < 1:
                        n1 += 1
                    if rank[i] < 2:
                        n2 += 1
                    if rank[i] < 3:
                        n3 += 1
            n = len(self.eval_df["key"].unique())
            acc1 = n1/n
            acc2 = n2/n
            acc3 = n3/n
            metrics = {"accuracy@1": acc1, "accuracy@2": acc2, "accuracy@3": acc3}
        else:
            EI_true, SN_true, TF_true, JP_true = [], [], [], []
            EI_pred, SN_pred, TF_pred, JP_pred = [], [], [], []
            self.eval_df.sort_values(by=["key", "attribute-name"], inplace=True)
            for key, sample_df in self.eval_df.groupby("key"):
                xlabel, _ = sample_df["label"]
                xpred, ypred = sample_df["pred"]
                xprob, yprob = sample_df["prob"]
                true = xlabel
                pred = np.nan
                if xpred == 1 and (ypred != 1 or xprob > yprob):
                    pred = 1
                elif ypred == 1 and (xpred != 1 or yprob > xprob):
                    pred = 0
                if pd.notna(pred):
                    if key.endswith("E/I"):
                        EI_true.append(true)
                        EI_pred.append(pred)
                    elif key.endswith("S/N"):
                        SN_true.append(true)
                        SN_pred.append(pred)
                    elif key.endswith("T/F"):
                        TF_true.append(true)
                        TF_pred.append(pred)
                    else:
                        JP_true.append(true)
                        JP_pred.append(pred)
            EI_f1 = f1_score(EI_true, EI_pred, average="binary") if len(EI_true) > 0 else 0
            SN_f1 = f1_score(SN_true, SN_pred, average="binary") if len(SN_true) > 0 else 0
            TF_f1 = f1_score(TF_true, TF_pred, average="binary") if len(TF_true) > 0 else 0
            JP_f1 = f1_score(JP_true, JP_pred, average="binary") if len(JP_true) > 0 else 0
            metrics = {"EI_f1": EI_f1, "SN_f1": SN_f1, "TF_f1": TF_f1, "JP_f1": JP_f1}
        return metrics

    def compute_instruction_metrics(self, evalprediction: EvalPrediction) -> Dict[str, float]:
        """Compute metrics for the instruction-tuning method"""
        labels = evalprediction.label_ids
        probs = evalprediction.predictions[:, :, 0]
        output_ids = evalprediction.predictions[:, :, 1]
        rx, cx = np.where(labels != -100)
        predictions = list(map(lambda x: x.strip().lower(),
                               self.tokenizer.batch_decode(output_ids[rx, cx - 1].reshape(-1, 1))))
        predictions = list(map(lambda x: 1 if x == "yes" else 0 if x == "no" else np.nan, predictions))
        probability = probs[rx, cx - 1].flatten()
        self.eval_df["pred"] = predictions
        self.eval_df["prob"] = probability
        return self._compute_metrics()

    def compute_classification_metrics(self, evalprediction: EvalPrediction) -> Dict[str, float]:
        """Compute metrics for the classification method"""
        logits = evalprediction.predictions
        probs = np.exp(logits)/np.sum(np.exp(logits), axis=-1, keepdims=True)
        self.eval_df["pred"] = np.argmax(probs, axis=-1)
        self.eval_df["prob"] = np.max(probs, axis=-1)
        return self._compute_metrics()

def finetune(_):
    """Instruction finetune or train binary classification LLMs on the character attribution classification task"""
    # decide experiments directory
    method_dir = "instruction" if FLAGS.instrtune else "classification"
    if FLAGS.contexts_file is not None:
        input_dir = FLAGS.contexts_file[:-6]
    else:
        input_dir = FLAGS.extracts_file[:-6]
    model_dir = FLAGS.model.replace("/", "--")
    if FLAGS.load_4bit:
        model_dir += "-4bit"
    elif FLAGS.load_8bit:
        model_dir += "-8bit"
    if FLAGS.bf16:
        model_dir += "-bf16"
    else:
        model_dir += "-fp16"
    datetime_dir = datetime.strftime(datetime.now(), "%Y%b%d-%H%M%S")
    experiments_dir = os.path.join(datadirs.datadir,
                                   "50-modeling/finetune",
                                   method_dir,
                                   input_dir,
                                   model_dir,
                                   datetime_dir)

    # set up logging
    logs_file = None
    if PARTIALSTATE.is_local_main_process and FLAGS.logtofile:
        os.makedirs(experiments_dir, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(program_name="finetune", log_dir=experiments_dir)
        logging.get_absl_handler().setFormatter(None)
        logs_file = os.path.join(experiments_dir, "logs.jsonl")

    # instantiate callback
    callback = LoggingCallback(logs_file)

    # log arguments
    if PARTIALSTATE.is_local_main_process:
        logging.info("ARGUMENTS")
        logging.info("================================================================================================")
        if FLAGS.contexts_file is not None:
            logging.info(f"{'contexts-file':30s} = {FLAGS.contexts_file}")
        else:
            logging.info(f"{'extracts-file':30s} = {FLAGS.extracts_file}")
        method = "instruction-tuning" if FLAGS.instrtune else "classification"
        logging.info(f"{'method':30s} = {method}")
        logging.info(f"{'model':30s} = {FLAGS.model}")
        precision = "bf16" if FLAGS.bf16 else "fp16"
        logging.info(f"{'precision':30s} = {precision}")
        quantization = "4bit" if FLAGS.load_4bit else "8bit" if FLAGS.load_8bit else "NONE"
        logging.info(f"{'quantization':30s} = {quantization}")
        logging.info(f"{'attention':30s} = {FLAGS.attn}")
        logging.info(f"{'train-batch-size':30s} = {FLAGS.train_batch_size}")
        logging.info(f"{'eval-batch-size':30s} = {FLAGS.eval_batch_size}")
        logging.info(f"{'eval-accumulation-steps':30s} = {FLAGS.eval_accumulation_steps}")
        logging.info(f"{'learning-rate':30s} = {FLAGS.lr}")
        logging.info(f"{'optimizer':30s} = {FLAGS.optim}")
        logging.info(f"{'weight-decay':30s} = {FLAGS.weight_decay}")
        logging.info(f"{'max-grad-norm':30s} = {FLAGS.max_grad_norm}")
        logging.info(f"{'metric':30s} = {FLAGS.metric}")
        logging.info(f"{'max-sequence-length':30s} = {FLAGS.max_seq_len}")
        logging.info(f"{'train-steps':30s} = {FLAGS.train_steps}")
        logging.info(f"{'eval-steps':30s} = {FLAGS.eval_steps}")
        logging.info(f"{'loggin-steps':30s} = {FLAGS.logging_steps}")
        logging.info(f"{'LoRA target-modules':30s} = {FLAGS.lora_target_module}")
        logging.info(f"{'LoRA rank':30s} = {FLAGS.rank}")
        logging.info(f"{'LoRA alpha':30s} = {FLAGS.alpha}")
        logging.info(f"{'LoRA dropout':30s} = {FLAGS.dropout}")
        logging.info("================================================================================================")
        logging.info("\n\n")

    # read data
    log("reading data")
    if FLAGS.contexts_file is not None:
        contexts_file = os.path.join(datadirs.datadir, "50-modeling/contexts", FLAGS.contexts_file)
        data = data_utils.load_contexts(contexts_file)
    else:
        extracts_file = os.path.join(datadirs.datadir, "50-modeling/extracts", FLAGS.extracts_file)
        data = data_utils.load_extracts(extracts_file)
    train_data = list(filter(lambda obj: obj["partition"] == "train", data))
    dev_data = list(filter(lambda obj: obj["partition"] == "dev", data))
    test_data = list(filter(lambda obj: obj["partition"] == "test", data))
    personet_data = data_utils.load_personet(test=True)
    personality_data = data_utils.load_story2personality()
    log(f"{len(train_data)} train examples")
    log(f"{len(dev_data)} dev examples")
    log(f"{len(test_data)} test examples\n\n")

    # print template
    if PARTIALSTATE.is_local_main_process:
        if FLAGS.instrtune:
            logging.info("TEMPLATE")
            logging.info("============================================================================================")
            logging.info(f"\"{TEMPLATE}\"")
            logging.info("============================================================================================")
            logging.info("\n")
            logging.info("ANSWER TEMPLATE")
            logging.info("============================================================================================")
            logging.info(f"\"{ANSWER_TEMPLATE}\"")
            logging.info("============================================================================================")
        else:
            logging.info(f"CHARACTER TOKEN = {CHARACTER_TOKEN}")
            logging.info(f"ATTRIBUTE TOKEN = {ATTRIBUTE_TOKEN}")
            logging.info(f"CONTEXT TOKEN   = {CONTEXT_TOKEN}")
        logging.info("\n\n")

    # instantiate quantization config
    compute_dtype = torch.bfloat16 if FLAGS.bf16 else torch.float16
    if FLAGS.load_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=compute_dtype,
                                                 bnb_4bit_quant_storage=compute_dtype,
                                                 bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_use_double_quant=True)
    elif FLAGS.load_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    # instantiate LoRA config
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM if FLAGS.instrtune else TaskType.SEQ_CLS,
                             target_modules=FLAGS.lora_target_module,
                             modules_to_save=["embed_tokens", "lm_head", "score"],
                             r=FLAGS.rank,
                             lora_alpha=FLAGS.alpha,
                             lora_dropout=FLAGS.dropout,
                             use_rslora=True,
                             bias="none")

    # instantiate model
    log("instantiating model")
    if FLAGS.instrtune:
        model = AutoModelForCausalLM.from_pretrained(FLAGS.model,
                                                     torch_dtype=compute_dtype,
                                                     quantization_config=quantization_config,
                                                     device_map={"": PARTIALSTATE.process_index},
                                                     attn_implementation=FLAGS.attn)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(FLAGS.model,
                                                                   num_labels=2,
                                                                   torch_dtype=compute_dtype,
                                                                   quantization_config=quantization_config,
                                                                   device_map={"": PARTIALSTATE.process_index},
                                                                   attn_implementation=FLAGS.attn)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id

    # instantiating tokenizer
    log("instantiating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # add special tokens (only done for classification method)
    if not FLAGS.instrtune:
        log("adding special tokens")
        tokenizer.add_special_tokens({"additional_special_tokens": [CHARACTER_TOKEN, ATTRIBUTE_TOKEN, CONTEXT_TOKEN]},
                                     replace_additional_special_tokens=False)
        model.resize_token_embeddings(len(tokenizer.vocab))

        # create LoRA model (only done for classification method)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        n_trainable, n_all = model.get_nb_trainable_parameters()
        log(f"{n_trainable} trainable params, {n_all} all params")

    # create datasets
    log("creating datasets")
    random.shuffle(train_data)
    train_dataset, _ = create_dataset(tokenizer, train_data)
    dev_dataset, dev_df = create_dataset(tokenizer, dev_data)
    test_dataset, test_df = create_dataset(tokenizer, test_data)
    personet_dataset, personet_df = create_dataset(tokenizer, personet_data)
    personality_dataset, personality_df = create_dataset(tokenizer, personality_data)
    train_ntokens = list(map(len, train_dataset["input_ids"]))
    dev_ntokens = list(map(len, dev_dataset["input_ids"]))
    test_ntokens = list(map(len, test_dataset["input_ids"]))
    personet_ntokens = list(map(len, personet_dataset["input_ids"]))
    personality_ntokens = list(map(len, personality_dataset["input_ids"]))
    log(f"CHATTER train tokens/sample: max = {max(train_ntokens)}, min = {min(train_ntokens)}, "
        f"95%tile = {np.quantile(train_ntokens, 0.95):.1f}")
    log(f"CHATTER dev tokens/sample: max = {max(dev_ntokens)}, min = {min(dev_ntokens)}, "
        f"95%tile = {np.quantile(dev_ntokens, 0.95):.1f}")
    log(f"CHATTER test tokens/sample: max = {max(test_ntokens)}, min = {min(test_ntokens)}, "
        f"95%tile = {np.quantile(test_ntokens, 0.95):.1f}")
    log(f"PERSONET test tokens/sample: max = {max(personet_ntokens)}, min = {min(personet_ntokens)}, "
        f"95%tile = {np.quantile(personet_ntokens, 0.95):.1f}")
    log(f"STORY2PERSONALITY tokens/sample: max = {max(personality_ntokens)}, min = {min(personality_ntokens)}, "
        f"95%tile = {np.quantile(personality_ntokens, 0.95):.1f}")

    # create SFT config
    if FLAGS.instrtune:
        config = SFTConfig(output_dir=experiments_dir,
                           eval_strategy="steps",
                           eval_steps=FLAGS.eval_steps,
                           eval_on_start=FLAGS.eval_on_start,
                           eval_accumulation_steps=FLAGS.eval_accumulation_steps,
                           per_device_train_batch_size=FLAGS.train_batch_size,
                           per_device_eval_batch_size=FLAGS.eval_batch_size,
                           learning_rate=FLAGS.lr,
                           weight_decay=FLAGS.weight_decay,
                           max_grad_norm=FLAGS.max_grad_norm,
                           max_steps=FLAGS.train_steps,
                           seed=2025,
                           data_seed=2025,
                           logging_strategy="steps",
                           logging_steps=FLAGS.logging_steps,
                           bf16=FLAGS.bf16,
                           fp16=not FLAGS.bf16,
                           optim=FLAGS.optim,
                           max_seq_length=FLAGS.max_seq_len,
                           packing=False,
                           save_strategy="best" if FLAGS.save_model else "no",
                           metric_for_best_model=f"test_{FLAGS.metric}",
                           save_only_model=True,
                           save_total_limit=1)
    else:
        config = TrainingArguments(output_dir=experiments_dir,
                                   eval_strategy="steps",
                                   eval_steps=FLAGS.eval_steps,
                                   eval_on_start=FLAGS.eval_on_start,
                                   eval_accumulation_steps=FLAGS.eval_accumulation_steps,
                                   per_device_train_batch_size=FLAGS.train_batch_size,
                                   per_device_eval_batch_size=FLAGS.eval_batch_size,
                                   learning_rate=FLAGS.lr,
                                   weight_decay=FLAGS.weight_decay,
                                   max_grad_norm=FLAGS.max_grad_norm,
                                   max_steps=FLAGS.train_steps,
                                   seed=2025,
                                   data_seed=2025,
                                   logging_strategy="steps",
                                   logging_steps=FLAGS.logging_steps,
                                   bf16=FLAGS.bf16,
                                   fp16=not FLAGS.bf16,
                                   optim=FLAGS.optim,
                                   gradient_checkpointing_kwargs={"use_reentrant": False})

    # create compute metrics instance
    compute_metrics = ComputeMetrics(tokenizer)
    compute_metrics.eval_df = dev_df
    compute_metrics.set_dataset("chatter")

    # create trainer
    log("instantiating trainer")
    if FLAGS.instrtune:
        trainer = SFTTrainer(model=model,
                             args=config,
                             data_collator=DataCollatorForCompletionOnlyLM(response_template=ANSWER_TEMPLATE,
                                                                           tokenizer=tokenizer),
                             train_dataset=train_dataset,
                             eval_dataset=dev_dataset,
                             processing_class=tokenizer,
                             peft_config=lora_config,
                             preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                             compute_metrics=compute_metrics.compute_instruction_metrics,
                             callbacks=[callback])
    else:
        trainer = Trainer(model=model,
                          args=config,
                          data_collator=DataCollatorWithPadding(tokenizer=tokenizer,
                                                                padding="longest",
                                                                return_tensors="pt"),
                          train_dataset=train_dataset,
                          eval_dataset=dev_dataset,
                          processing_class=tokenizer,
                          compute_metrics=compute_metrics.compute_classification_metrics,
                          callbacks=[callback])

    # train
    log("\n\n\ntraining started")
    log("================================================================================================")
    trainer.train()
    log("\n\n\n==========================================================================================")
    log("training done\n\n\n")

    # predict and evaluate
    log("evaluating CHATTER dev set")
    dev_output = trainer.predict(dev_dataset)
    log(f"{dev_output.metrics}\n\n")

    log("evaluating CHATTER test set")
    compute_metrics.eval_df = test_df
    test_output = trainer.predict(test_dataset)
    log(f"{test_output.metrics}\n\n")

    log("evaluating PERSONET test set")
    compute_metrics.eval_df = personet_df
    compute_metrics.set_dataset("personet")
    personet_output = trainer.predict(personet_dataset)
    log(f"{personet_output.metrics}\n\n")

    log("evaluating STORY2PERSONALITY dataset")
    compute_metrics.eval_df = personality_df
    compute_metrics.set_dataset("personality")
    config.per_device_eval_batch_size = 1
    personality_output = trainer.predict(personality_dataset)
    log(f"{personality_output.metrics}")

    if PARTIALSTATE.is_local_main_process and FLAGS.save_predictions:
        dev_file = os.path.join(experiments_dir, "CHATTER_dev.csv")
        dev_metrics_file = os.path.join(experiments_dir, "CHATTER_dev.json")
        dev_df.to_csv(dev_file, index=False)
        json.dump(dev_output.metrics, open(dev_metrics_file, "w"))
        test_file = os.path.join(experiments_dir, "CHATTER_test.csv")
        test_metrics_file = os.path.join(experiments_dir, "CHATTER_test.json")
        test_df.to_csv(test_file, index=False)
        json.dump(test_output.metrics, open(test_metrics_file, "w"))
        personet_file = os.path.join(experiments_dir, "PERSONET.csv")
        personet_metrics_file = os.path.join(experiments_dir, "PERSONET.json")
        personet_df.to_csv(personet_file, index=False)
        json.dump(personet_output.metrics, open(personet_metrics_file, "w"))
        personality_file = os.path.join(experiments_dir, "PERSONALITY.csv")
        personality_metrics_file = os.path.join(experiments_dir, "PERSONALITY.json")
        personality_df.to_csv(personality_file, index=False)
        json.dump(personality_output.metrics, open(personality_metrics_file, "w"))

    # remove empty experiments dir
    try:
        os.rmdir(experiments_dir)
        log(f"deleting {experiments_dir}")
    except OSError:
        pass

if __name__ == '__main__':
    app.run(finetune)