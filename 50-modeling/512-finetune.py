"""Instruction Fine-tune LLMs on the character attribution classification task"""
import datadirs

from absl import app
from absl import flags
from absl import logging
from accelerate import PartialState
import collections
from datasets import Dataset
import jsonlines
import numpy as np
import os
import pandas as pd
from peft import LoraConfig, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer import EvalPrediction
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from typing import Dict

flags.DEFINE_string("data_file", default=None, help="jsonlines file of samples")
flags.DEFINE_string("train_file", default=None, help="jsonlines file of training samples")
flags.DEFINE_string("dev_file", default=None, help="jsonlines file of dev samples")
flags.DEFINE_string("test_file", default=None, help="jsonlines file of test samples")
flags.DEFINE_string("model", default="meta-llama/Llama-3.1-8B-Instruct", help="huggingface model name")
flags.DEFINE_bool("bf16", default=False, help="use brain floating point (default=fp16)")
flags.DEFINE_bool("load_4bit", default=False, help="do 4-bit QLoRA training")
flags.DEFINE_bool("load_8bit", default=False, help="do 8-bit QLoRA training")
flags.DEFINE_bool("flash_attn", default=False, help="use flash attention (default=scaled dot product 'sdpa')")
flags.DEFINE_integer("dataset_batch_size", default=1024, help="dataset batch size for tokenization")
flags.DEFINE_integer("train_batch_size", default=1, help="training batch size")
flags.DEFINE_integer("eval_batch_size", default=1, help="evaluation batch size")
flags.DEFINE_integer("eval_accumulation_steps", default=1, help=("number of prediction steps to accumulate the output "
                                                                 "tensors for, before moving to CPU"))
flags.DEFINE_float("lr", default=1e-5, help="learning rate")
flags.DEFINE_string("optim", default="adamw_torch", help=("optimizer name (https://github.com/huggingface/"
                                                          "transformers/blob/main/src/transformers/training_args.py)"))
flags.DEFINE_integer("max_seq_len", default=1024, help="maximum sequence length")
flags.DEFINE_integer("train_steps", default=1024, help="training steps")
flags.DEFINE_integer("eval_steps", default=64, help="training steps between successive evaluations")
flags.DEFINE_integer("logging_steps", default=8, help="training steps between successive logging")
flags.DEFINE_bool("eval_testset", default=False, help="evaluate test set every eval_steps (default=dev set)")
flags.DEFINE_multi_string("lora_target_module", default=["q_proj", "k_proj"], help="target modules to train using LoRA")
flags.DEFINE_multi_string("lora_save_module", default=["lm_head", "embed_tokens"], help="modules to train normally")
flags.DEFINE_integer("rank", default=8, help="lora rank")
flags.DEFINE_integer("alpha", default=16, help="lora alpha")
flags.DEFINE_float("dropout", default=0, help="dropout")
flags.DEFINE_float("weight_decay", default=0, help="weight decay")
flags.DEFINE_float("max_grad_norm", default=10, help="maximum gradient norm")
flags.DEFINE_enum("metric", default="f1", enum_values=["accuracy", "f1", "precision", "recall"],
                  help="metric to compare models")
flags.DEFINE_bool("logtofile", default=False, help="log to file")
flags.DEFINE_bool("save_predictions", default=False,
                  help="save predictions of eval set corresponding to the best metric value")

def input_checker(args):
    given_data = args["data_file"] is not None
    given_train = args["train_file"] is not None
    return (given_data and not given_train) or (not given_data and given_train)

def dev_checker(args):
    given_train = args["train_file"] is not None
    given_dev = args["dev_file"] is not None
    return not given_train or given_dev

flags.register_multi_flags_validator(["data_file", "train_file"], input_checker,
                                     "Provide exactly one of data_file or train_file")
flags.register_multi_flags_validator(["train_file", "dev_file"], dev_checker, "Provide dev_file")

FLAGS = flags.FLAGS

TEMPLATE = ("Given the definition of a character attribute or trope, the name of a character, and a story or segments "
            "of a story where the character appears, speaks or is mentioned, answer 'yes' or 'no' if the character "
            "portrays or is associated with the attribute or trope in the story.\n\nATTRIBUTE: $ATTRIBUTE$"
            "\n\nCHARACTER: $CHARACTER$\n\nSTORY: $STORY$ \n\n ANSWER: $ANSWER$")
ANSWER_TEMPLATE = " \n\n ANSWER:"

class LoggingCallback(TrainerCallback):
    def __init__(self, predictions_file):
        super().__init__()
        self.predictions_file = predictions_file
        self.eval_df = None
        self.best_metric_value = None

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float],
               **kwargs):
        if state.is_local_process_zero:
            logging.info(f"STEP {state.global_step}/{state.max_steps}")
            for logkey, logvalue in logs.items():
                logging.info(f"{logkey} = {logvalue:.6f}")
            logging.info("\n")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_local_process_zero and FLAGS.save_predictions and (
            self.best_metric_value is None or state.best_metric > self.best_metric_value):
            self.best_metric_value = state.best_metric
            logging.info(f"STEP {state.global_step}/{state.max_steps}")
            logging.info("saving predictions")
            self.eval_df.to_csv(self.predictions_file)
            logging.info("\n")

def create_dataset(tokenizer, data, batch_size, max_seq_len):
    texts = []
    rows = []
    for obj in data:
        text = (TEMPLATE
                .replace("$ANSWER$", "yes" if obj["label"] == 1 else "no")
                .replace("$CHARACTER$", obj["name"])
                .replace("$ATTRIBUTE$", obj["definition"])
                .replace("$STORY$", obj["text"])
                )
        row = [obj["character"], obj["trope"], obj["label"]]
        rows.append(row)
        texts.append(text)
    df = pd.DataFrame(rows, columns=["character", "trope", "label"])
    input_ids, attention_mask = [], []
    n_batches = int(np.ceil(len(texts)/batch_size))
    for i in tqdm.trange(n_batches, desc="tokenization"):
        batch_texts = texts[batch_size * i: batch_size * (i + 1)]
        encoding = tokenizer(batch_texts, padding=False, add_special_tokens=True, return_overflowing_tokens=False,
                             return_length=False)
        input_ids.extend(encoding["input_ids"])
        attention_mask.extend(encoding["attention_mask"])
    yes_answer_sp_mask = tokenizer(ANSWER_TEMPLATE + " yes", add_special_tokens=True,
                                   return_special_tokens_mask=True)["special_tokens_mask"]
    no_answer_sp_mask = tokenizer(ANSWER_TEMPLATE + " no", add_special_tokens=True,
                                  return_special_tokens_mask=True)["special_tokens_mask"]
    is_first_token_sp = yes_answer_sp_mask[0] == 1
    is_last_token_sp = yes_answer_sp_mask[-1] == 1
    yes_answer_size = len(yes_answer_sp_mask) - is_first_token_sp - is_last_token_sp
    no_answer_size = len(no_answer_sp_mask) - is_first_token_sp - is_last_token_sp
    for i in tqdm.trange(len(input_ids), desc="truncation"):
        if len(input_ids[i]) > max_seq_len:
            if data[i]["label"] == 1:
                start = -is_last_token_sp - yes_answer_size - (len(input_ids[i]) - max_seq_len)
                end = -is_last_token_sp - yes_answer_size
            else:
                start = -is_last_token_sp - no_answer_size - (len(input_ids[i]) - max_seq_len)
                end = -is_last_token_sp - no_answer_size
            input_ids[i] = input_ids[i][:start] + input_ids[i][end:]
            attention_mask[i] = attention_mask[i][:start] + attention_mask[i][end:]
    return Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask}), df

def preprocess_logits_for_metrics(logits, _):
    return logits.argmax(dim=-1)

def compute_metrics(tokenizer, eval_df, callback: LoggingCallback, evalprediction: EvalPrediction):
    labels = evalprediction.label_ids
    predictions = evalprediction.predictions
    rx, cx = np.where(labels != -100)
    predictions = list(map(lambda x: x.strip().lower(), tokenizer.batch_decode(predictions[rx, cx - 1].reshape(-1, 1))))
    predictions = list(map(lambda x: 1 if x == "yes" else 0 if x == "no" else np.nan, predictions))
    eval_df["pred"] = predictions
    eval_df = eval_df.dropna(subset="pred")
    callback.eval_df = eval_df
    n_samples = len(eval_df[["character", "trope"]].drop_duplicates())
    if len(eval_df) > 0:
        eval_group_df = eval_df.groupby(["character", "trope"]).agg({"label": lambda arr: arr.values[0],
                                                                     "pred": lambda arr: int((arr == 1).any())})
        n_eval_samples = len(eval_group_df)
        labels = eval_group_df["label"].tolist()
        predictions = eval_group_df["pred"].tolist()
        acc = accuracy_score(labels, predictions)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    else:
        acc, prec, rec, f1, n_eval_samples = 0, 0, 0, 0, 0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "total_samples": n_samples,
            "evaluated_samples": n_eval_samples}

def finetune(_):
    """Instruction Fine-tune LLMs on the character attribution classification task"""
    # instantiate Partial State
    partial_state = PartialState()

    # set up logging
    if FLAGS.data_file is not None:
        output_dir = FLAGS.data_file.replace("/", "-")[:-6]
    else:
        output_dir = FLAGS.train_file.replace("/", "-")[:-6]
    experiments_dir = os.path.join(datadirs.datadir, "50-modeling/finetune", output_dir)
    if FLAGS.logtofile and partial_state.is_local_main_process:
        os.makedirs(experiments_dir, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(program_name="finetune", log_dir=experiments_dir)
        logging.get_absl_handler().setFormatter(None)

    # log arguments
    if partial_state.is_local_main_process:
        logging.info("ARGUMENTS")
        logging.info("================================================================================================")
        if FLAGS.data_file is not None:
            logging.info(f"{'data-file':30s} = {FLAGS.data_file}")
        else:
            logging.info(f"{'train-file':30s} = {FLAGS.train_file}")
            logging.info(f"{'dev-file':30s} = {FLAGS.dev_file}")
            logging.info(f"{'test-file':30s} = {FLAGS.test_file}")
        logging.info(f"{'model':30s} = {FLAGS.model}")
        precision = "bf16" if FLAGS.bf16 else "fp16"
        logging.info(f"{'precision':30s} = {precision}")
        quantization = "4bit" if FLAGS.load_4bit else "8bit" if FLAGS.load_8bit else "NONE"
        logging.info(f"{'quantization':30s} = {quantization}")
        attn_implementation = "flash_attn2" if FLAGS.flash_attn else "sdpa"
        logging.info(f"{'attention':30s} = {attn_implementation}")
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
        evalset = "test-set" if FLAGS.eval_testset else "dev-set"
        logging.info(f"{'eval-set':30s} = {evalset}")
        logging.info(f"{'LoRA target-modules':30s} = {FLAGS.lora_target_module}")
        logging.info(f"{'LoRA save-modules':30s} = {FLAGS.lora_save_module}")
        logging.info(f"{'LoRA rank':30s} = {FLAGS.rank}")
        logging.info(f"{'LoRA alpha':30s} = {FLAGS.alpha}")
        logging.info(f"{'LoRA dropout':30s} = {FLAGS.dropout}")
        logging.info("================================================================================================")
        logging.info("\n\n")

    # read data
    if partial_state.is_local_main_process:
        logging.info("reading data")
    if FLAGS.data_file is not None:
        data_file = os.path.join(datadirs.datadir, "50-modeling", FLAGS.data_file)
        with jsonlines.open(data_file) as reader:
            data = list(reader)
        if hasattr(data[0], "partition"):
            train_data = [sample for sample in data if sample["partition"] == "train"]
            dev_data = [sample for sample in data if sample["partition"] == "dev"]
            test_data = [sample for sample in data if sample["partition"] == "test"]
        else:
            label_file = os.path.join(datadirs.datadir, "CHATTER/chatter.csv")
            tropes_file = os.path.join(datadirs.datadir, "CHATTER/tropes.csv")
            label_df = pd.read_csv(label_file, index_col=None)
            tropes_df = pd.read_csv(tropes_file, index_col=None)
            characterid_to_ixs = collections.defaultdict(list)
            trope_to_definition = {}
            train_data, dev_data, test_data = [], [], []
            for i, obj in enumerate(data):
                characterid = obj["character"]
                characterid_to_ixs[characterid].append(i)
            for _, row in tropes_df.iterrows():
                trope_to_definition[row["trope"]] = row["summary"]
            for _, row in label_df[label_df["partition"].notna()].iterrows():
                characterid, trope, partition = row["character"], row["trope"], row["partition"]
                definition = trope_to_definition[trope]
                label = row["label"] if partition == "test" else row["tvtrope-label"]
                for i in characterid_to_ixs[characterid]:
                    obj = data[i]
                    sample = {"text": obj["text"], "imdbid": obj["imdbid"], "character": characterid,
                              "trope": trope, "definition": definition, "name": obj["name"], "label": label}
                    if partition == "train":
                        train_data.append(sample)
                    elif partition == "dev":
                        dev_data.append(sample)
                    else:
                        test_data.append(sample)
    else:
        train_file = os.path.join(datadirs.datadir, "50-modeling", FLAGS.train_file)
        dev_file = os.path.join(datadirs.datadir, "50-modeling", FLAGS.dev_file)
        test_file = os.path.join(datadirs.datadir, "50-modeling", FLAGS.test_file)
        with jsonlines.open(train_file) as reader:
            train_data = list(reader)
        with jsonlines.open(dev_file) as reader:
            dev_data = list(reader)
        with jsonlines.open(test_file) as reader:
            test_data = list(reader)
    if partial_state.is_local_main_process:
        logging.info(f"{len(train_data)} train examples")
        logging.info(f"{len(dev_data)} dev examples")
        logging.info(f"{len(test_data)} test examples\n\n")

    # print template
    if partial_state.is_local_main_process:
        logging.info("TEMPLATE")
        logging.info("================================================================================================")
        logging.info(f"\"{TEMPLATE}\"")
        logging.info("================================================================================================")
        logging.info("\n")
        logging.info("ANSWER TEMPLATE")
        logging.info("================================================================================================")
        logging.info(f"\"{ANSWER_TEMPLATE}\"")
        logging.info("================================================================================================")
        logging.info("\n\n")

    # instantiate model
    if partial_state.is_local_main_process:
        logging.info("instantiating model")
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
    model = AutoModelForCausalLM.from_pretrained(FLAGS.model,
                                                 torch_dtype=compute_dtype,
                                                 quantization_config=quantization_config,
                                                 device_map={"": partial_state.process_index},
                                                 attn_implementation=("flash_attention_2" if FLAGS.flash_attn else 
                                                                      "sdpa"))

    # instantiating tokenizer
    if partial_state.is_local_main_process:
        logging.info("instantiating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # create datasets
    if partial_state.is_local_main_process:
        logging.info("creating datasets")
    train_dataset, _ = create_dataset(tokenizer, train_data, FLAGS.dataset_batch_size, FLAGS.max_seq_len)
    dev_dataset, dev_df = create_dataset(tokenizer, dev_data, FLAGS.dataset_batch_size, FLAGS.max_seq_len)
    test_dataset, test_df = create_dataset(tokenizer, test_data, FLAGS.dataset_batch_size, FLAGS.max_seq_len)
    if FLAGS.eval_testset:
        eval_dataset, eval_df = test_dataset, test_df
    else:
        eval_dataset, eval_df = dev_dataset, dev_df

    # create peft config
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             target_modules=FLAGS.lora_target_module,
                             modules_to_save=FLAGS.lora_save_module,
                             r=FLAGS.rank,
                             lora_alpha=FLAGS.alpha,
                             lora_dropout=FLAGS.dropout,
                             use_rslora=True,
                             bias="none")

    # create SFT config
    config = SFTConfig(output_dir=experiments_dir,
                       eval_strategy="steps",
                       eval_steps=FLAGS.eval_steps,
                       eval_on_start=True,
                       eval_accumulation_steps=FLAGS.eval_accumulation_steps,
                       per_device_train_batch_size=FLAGS.train_batch_size,
                       per_device_eval_batch_size=FLAGS.eval_batch_size,
                       learning_rate=FLAGS.lr,
                       weight_decay=FLAGS.weight_decay,
                       max_grad_norm=FLAGS.max_grad_norm,
                       max_steps=FLAGS.train_steps,
                       logging_strategy="steps",
                       logging_steps=FLAGS.logging_steps,
                       save_strategy="no",
                       bf16=FLAGS.bf16,
                       fp16=not FLAGS.bf16,
                       optim=FLAGS.optim,
                       max_seq_length=FLAGS.max_seq_len,
                       packing=False,
                       metric_for_best_model=FLAGS.metric)

    # create trainer
    if partial_state.is_local_main_process:
        logging.info("instantiating trainer")
    predictions_file = os.path.join(experiments_dir, f"{os.getpid()}-predictions.csv")
    callback = LoggingCallback(predictions_file)
    trainer = SFTTrainer(model=model,
                         args=config,
                         data_collator=DataCollatorForCompletionOnlyLM(response_template=ANSWER_TEMPLATE,
                                                                       tokenizer=tokenizer),
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         processing_class=tokenizer,
                         peft_config=lora_config,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                         compute_metrics=lambda x: compute_metrics(tokenizer, eval_df, callback, x),
                         callbacks=[callback])

    # train
    if partial_state.is_local_main_process:
        logging.info("\n\ntraining")
        logging.info("================================================================================================")
    trainer.train()

if __name__ == '__main__':
    app.run(finetune)