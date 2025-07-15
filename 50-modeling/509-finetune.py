"""Instruction Fine-tune LLMs on the character attribution classification task"""
import data_utils
import eval_utils

from absl import app
from absl import flags
from absl import logging
from accelerate import PartialState
from datetime import datetime
import json
import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
import random
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from typing import Dict

flags.DEFINE_enum("train_dataset", default="chatter", enum_values=["chatter", "personet"], help="training dataset")
flags.DEFINE_string("contexts_file", default=None, help="contexts data file")
flags.DEFINE_string("extracts_file", default=None, help="extracts data file")
flags.DEFINE_bool("anonymize", default=False, help="use anonymized contexts or extracts for training")
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
flags.DEFINE_bool("eval", default=True, help="do evaluation during training")
flags.DEFINE_integer("eval_batch_size", default=1, help="evaluation batch size")
flags.DEFINE_integer("eval_accumulation_steps",
                     default=None,
                     help="number of prediction steps to accumulate the output tensors for, before moving to CPU")
flags.DEFINE_integer("eval_delay",
                     default=None,
                     help="Number of epochs or steps to wait for before the first evaluation can be performed")
flags.DEFINE_float("lr", default=2e-5, help="learning rate")
flags.DEFINE_integer("warmup_steps", default=0, help="number of steps used for linear warmup from 0 to lr")
flags.DEFINE_string("optim",
                    default="adamw_torch",
                    help=("optimizer name (https://github.com/huggingface/transformers/blob/main/src/transformers/"
                          "training_args.py)"))
flags.DEFINE_integer("chatter_instr_seqlen", default=1800, help="sequence length for chatter instructions")
flags.DEFINE_integer("personet_instr_seqlen", default=1700, help="sequence length for personet instructions")
flags.DEFINE_integer("train_steps", default=1024, help="training steps")
flags.DEFINE_integer("eval_steps", default=32, help="training steps between successive evaluations")
flags.DEFINE_integer("logging_steps", default=8, help="training steps between successive logging")
flags.DEFINE_multi_string("lora_target_module", default=["q_proj", "k_proj"], help="target modules to train using LoRA")
flags.DEFINE_integer("rank", default=32, help="lora rank")
flags.DEFINE_integer("alpha", default=64, help="lora alpha")
flags.DEFINE_float("dropout", default=0, help="dropout")
flags.DEFINE_float("weight_decay", default=0, help="weight decay")
flags.DEFINE_float("max_grad_norm", default=10, help="maximum gradient norm")
flags.DEFINE_bool("save_model", default=False, help="save model safetensors at the end of training")

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

# set seeds
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

# logging utility
def log(message):
    if PARTIALSTATE.is_local_main_process:
        logging.info(message)

# define logging callback class
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

def preprocess_logits_for_metrics(logits: torch.Tensor, _) -> torch.Tensor:
    """Preprocess logits to prepare for metrics computation for the instruction-tuning method.
    We use this functions to save space primarily, converting batch-size x seqlen x vocab-size tensors to batch-size
    x seqlen x 2 tensors
    """
    probs = logits.softmax(dim=-1).float() # batch-size x seqlen x vocab-size
    max_output = torch.max(probs, dim=-1)
    z = torch.cat((max_output.values.unsqueeze(dim=-1),
                   max_output.indices.unsqueeze(dim=-1)), dim=-1) # batch-size x seqlen x 2
    return z

def plot_logs(logs, metric, experiments_dir):
    train_loss, train_steps = [], []
    dev_loss, dev_metric, dev_steps = [], [], []
    rows = []
    for log in logs:
        if "loss" in log:
            train_loss.append(log["loss"])
            train_steps.append(log["step"])
            rows.append([log["step"], "train", log["loss"], ""])
        elif "eval_loss" in log:
            dev_loss.append(log["eval_loss"])
            dev_metric.append(log[metric])
            dev_steps.append(log["step"])
            rows.append([log["step"], "dev", log["eval_loss"], log[metric]])
    plt.figure(figsize=(25, 12))
    plt.plot(train_steps, train_loss, color="blue", lw=5, marker="o", ms=15, label="train loss")
    plt.plot(dev_steps, dev_loss, color="red", lw=5, marker="s", ms=15, label="dev loss")
    plt.plot(dev_steps, dev_metric, color="green", lw=5, marker="^", ms=15, label=f"dev {metric}")
    plt.ylabel("metric")
    plt.xlabel("step")
    plt.legend(fontsize="x-large")
    plots_file = os.path.join(experiments_dir, "plot.png")
    progress_file = os.path.join(experiments_dir, "progress.csv")
    plt.savefig(plots_file)
    df = pd.DataFrame(rows, columns=["step", "partition", "loss", metric])
    df.to_csv(progress_file, index=False)

def finetune(_):
    """Instruction finetune or train binary classification LLMs on the character attribution classification task"""
    # decide experiments directory
    method_dir = "instruction" if FLAGS.instrtune else "classification"
    if FLAGS.train_dataset == "chatter":
        if FLAGS.contexts_file is not None:
            input_dir = FLAGS.contexts_file[:-6]
        else:
            input_dir = FLAGS.extracts_file[:-6]
        if FLAGS.anonymize:
            input_dir = input_dir + "-anonymized"
    else:
        input_dir = "personet"
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
    experiments_dir = os.path.join(data_utils.DATADIR,
                                   "50-modeling/finetune",
                                   method_dir,
                                   input_dir,
                                   model_dir,
                                   datetime_dir)

    # set up logging
    logs_file = None
    if PARTIALSTATE.is_local_main_process:
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
        logging.info(f"{'train dataset':30s} = {FLAGS.train_dataset}")
        if FLAGS.contexts_file is not None:
            logging.info(f"{'contexts-file':30s} = {FLAGS.contexts_file}")
        else:
            logging.info(f"{'extracts-file':30s} = {FLAGS.extracts_file}")
        logging.info(f"{'anonymize':30s} = {FLAGS.anonymize}")
        method = "instruction-tuning" if FLAGS.instrtune else "classification"
        logging.info(f"{'method':30s} = {method}")
        logging.info(f"{'model':30s} = {FLAGS.model}")
        precision = "bf16" if FLAGS.bf16 else "fp16"
        logging.info(f"{'precision':30s} = {precision}")
        quantization = "4bit" if FLAGS.load_4bit else "8bit" if FLAGS.load_8bit else "NONE"
        logging.info(f"{'quantization':30s} = {quantization}")
        logging.info(f"{'attention':30s} = {FLAGS.attn}")
        logging.info(f"{'train-batch-size':30s} = {FLAGS.train_batch_size}")
        logging.info(f"{'do-eval':30s} = {FLAGS.eval}")
        logging.info(f"{'eval-batch-size':30s} = {FLAGS.eval_batch_size}")
        logging.info(f"{'eval-accumulation-steps':30s} = {FLAGS.eval_accumulation_steps}")
        logging.info(f"{'eval-delay':30s} = {FLAGS.eval_delay}")
        logging.info(f"{'learning-rate':30s} = {FLAGS.lr}")
        logging.info(f"{'warmup-steps':30s} = {FLAGS.warmup_steps}")
        logging.info(f"{'optimizer':30s} = {FLAGS.optim}")
        logging.info(f"{'CHATTER instruction length':30s} = {FLAGS.chatter_instr_seqlen}")
        logging.info(f"{'PERSONET instruction length':30s} = {FLAGS.personet_instr_seqlen}")
        logging.info(f"{'weight-decay':30s} = {FLAGS.weight_decay}")
        logging.info(f"{'max-grad-norm':30s} = {FLAGS.max_grad_norm}")
        logging.info(f"{'train-steps':30s} = {FLAGS.train_steps}")
        logging.info(f"{'eval-steps':30s} = {FLAGS.eval_steps}")
        logging.info(f"{'logging-steps':30s} = {FLAGS.logging_steps}")
        logging.info(f"{'LoRA target-modules':30s} = {FLAGS.lora_target_module}")
        logging.info(f"{'LoRA rank':30s} = {FLAGS.rank}")
        logging.info(f"{'LoRA alpha':30s} = {FLAGS.alpha}")
        logging.info(f"{'LoRA dropout':30s} = {FLAGS.dropout}")
        logging.info("================================================================================================")
        logging.info("\n\n")

    # read data
    log("reading chatter data")
    if FLAGS.contexts_file is not None:
        contexts_file = os.path.join(data_utils.DATADIR, "50-modeling/contexts", FLAGS.contexts_file)
        anonymized_contexts_file = os.path.join(data_utils.DATADIR,
                                                "50-modeling/anonymized-contexts",
                                                FLAGS.contexts_file)
        chatter_data = data_utils.load_contexts(contexts_file)
        anonymized_chatter_data = data_utils.load_contexts(anonymized_contexts_file, anonymize=True)
    else:
        extracts_file = os.path.join(data_utils.DATADIR, "50-modeling/extracts", FLAGS.extracts_file)
        anonymized_extracts_file = os.path.join(data_utils.DATADIR,
                                                "50-modeling/anonymized-extracts",
                                                FLAGS.extracts_file)
        chatter_data = data_utils.load_extracts(extracts_file)
        anonymized_chatter_data = data_utils.load_extracts(anonymized_extracts_file, anonymize=True)
    log("reading personet data")
    personet_data = data_utils.load_personet()
    chatter_train_data = list(filter(lambda obj: obj["partition"] == "train", chatter_data))
    chatter_dev_data = list(filter(lambda obj: obj["partition"] == "dev", chatter_data))
    chatter_test_data = list(filter(lambda obj: obj["partition"] == "test", chatter_data))
    anonymized_chatter_train_data = list(filter(lambda obj: obj["partition"] == "train", anonymized_chatter_data))
    anonymized_chatter_dev_data = list(filter(lambda obj: obj["partition"] == "dev", anonymized_chatter_data))
    anonymized_chatter_test_data = list(filter(lambda obj: obj["partition"] == "test", anonymized_chatter_data))
    personet_train_data = list(filter(lambda obj: obj["partition"] == "train", personet_data))
    personet_dev_data = list(filter(lambda obj: obj["partition"] == "dev", personet_data))
    personet_test_data = list(filter(lambda obj: obj["partition"] == "test", personet_data))
    if PARTIALSTATE.is_local_main_process:
        logging.info(f"{len(chatter_train_data)} CHATTER train examples")
        logging.info(f"{len(chatter_dev_data)} CHATTER dev examples")
        logging.info(f"{len(chatter_test_data)} CHATTER test examples")
        logging.info(f"{len(anonymized_chatter_train_data)} ANONYMIZED CHATTER train examples")
        logging.info(f"{len(anonymized_chatter_dev_data)} ANONYMIZED CHATTER dev examples")
        logging.info(f"{len(anonymized_chatter_test_data)} ANONYMIZED CHATTER test examples")
        logging.info(f"{len(personet_train_data)} PERSONET train examples")
        logging.info(f"{len(personet_dev_data)} PERSONET dev examples")
        logging.info(f"{len(personet_test_data)} PERSONET test examples")
        logging.info("\n\n")

    # assign train data
    if FLAGS.train_dataset == "chatter":
        if FLAGS.anonymize:
            train_data = anonymized_chatter_train_data
        else:
            train_data = chatter_train_data
        train_instr_seqlen = FLAGS.chatter_instr_seqlen
    else:
        train_data = personet_train_data
        train_instr_seqlen = FLAGS.personet_instr_seqlen

    # print template
    if PARTIALSTATE.is_local_main_process:
        if FLAGS.instrtune:
            logging.info("TEMPLATE")
            logging.info("============================================================================================")
            logging.info(f"\"{data_utils.TEMPLATE}\"")
            logging.info("============================================================================================")
            logging.info("\n")
            logging.info("ANSWER TEMPLATE")
            logging.info("============================================================================================")
            logging.info(f"\"{data_utils.ANSWER_TEMPLATE}\"")
            logging.info("============================================================================================")
        else:
            logging.info(f"CHARACTER TOKEN = {data_utils.CHARACTER_TOKEN}")
            logging.info(f"ATTRIBUTE TOKEN = {data_utils.ATTRIBUTE_TOKEN}")
            logging.info(f"CONTEXT TOKEN   = {data_utils.CONTEXT_TOKEN}")
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
    log("\n\ninstantiating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # add special tokens (only done for classification method)
    if not FLAGS.instrtune:
        log("adding special tokens")
        tokenizer.add_special_tokens({"additional_special_tokens": [data_utils.CHARACTER_TOKEN,
                                                                    data_utils.ATTRIBUTE_TOKEN,
                                                                    data_utils.CONTEXT_TOKEN]},
                                     replace_additional_special_tokens=False)
        model.resize_token_embeddings(len(tokenizer.vocab))

        # create LoRA model (only done for classification method)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        n_trainable, n_all = model.get_nb_trainable_parameters()
        log(f"{n_trainable} trainable params, {n_all} all params, {100*n_trainable/n_all:.1f}%trainable")
    log("\n\n")

    # create datasets
    log("creating datasets")
    random.shuffle(train_data)
    kwargs = dict(tokenizer=tokenizer,
                  instrtune=FLAGS.instrtune,
                  batch_size=FLAGS.dataset_batch_size,
                  disable_progress_bar=not PARTIALSTATE.is_local_main_process)
    train_dataset, _ = data_utils.create_dataset(data=train_data,
                                                 name="train",
                                                 instr_seqlen=train_instr_seqlen,
                                                 **kwargs)
    chatter_dev_dataset, chatter_dev_df = data_utils.create_dataset(data=chatter_dev_data,
                                                                    name="chatter dev",
                                                                    instr_seqlen=FLAGS.chatter_instr_seqlen,
                                                                    **kwargs)
    chatter_test_dataset, chatter_test_df = data_utils.create_dataset(data=chatter_test_data,
                                                                      name="chatter test",
                                                                      instr_seqlen=FLAGS.chatter_instr_seqlen,
                                                                      **kwargs)
    anonymized_chatter_dev_dataset, anonymized_chatter_dev_df = data_utils.create_dataset(
        data=anonymized_chatter_dev_data,
        name="anonymized chatter dev",
        instr_seqlen=FLAGS.chatter_instr_seqlen,
        **kwargs)
    anonymized_chatter_test_dataset, anonymized_chatter_test_df = data_utils.create_dataset(
        data=anonymized_chatter_test_data,
        name="anonymized chatter test",
        instr_seqlen=FLAGS.chatter_instr_seqlen,
        **kwargs)
    personet_dev_dataset, personet_dev_df = data_utils.create_dataset(data=personet_dev_data,
                                                                      name="personet dev",
                                                                      instr_seqlen=FLAGS.personet_instr_seqlen,
                                                                      **kwargs)
    personet_test_dataset, personet_test_df = data_utils.create_dataset(data=personet_test_data,
                                                                        name="personet test",
                                                                        instr_seqlen=FLAGS.personet_instr_seqlen,
                                                                        **kwargs)
    train_ntokens = list(map(len, train_dataset["input_ids"]))
    chatter_dev_ntokens = list(map(len, chatter_dev_dataset["input_ids"]))
    chatter_test_ntokens = list(map(len, chatter_test_dataset["input_ids"]))
    anonymized_chatter_dev_ntokens = list(map(len, anonymized_chatter_dev_dataset["input_ids"]))
    anonymized_chatter_test_ntokens = list(map(len, anonymized_chatter_test_dataset["input_ids"]))
    personet_dev_ntokens = list(map(len, personet_dev_dataset["input_ids"]))
    personet_test_ntokens = list(map(len, personet_test_dataset["input_ids"]))
    if PARTIALSTATE.is_local_main_process:
        logging.info("\ntokens/sample:")
        logging.info(f"{FLAGS.train_dataset.upper()} train: max = {max(train_ntokens)}, min = {min(train_ntokens)}, "
                     f"95%tile = {np.quantile(train_ntokens, 0.95):.1f}")
        logging.info(f"CHATTER dev: max = {max(chatter_dev_ntokens)}, min = {min(chatter_dev_ntokens)}, "
                     f"95%tile = {np.quantile(chatter_dev_ntokens, 0.95):.1f}")
        logging.info(f"CHATTER test: max = {max(chatter_test_ntokens)}, min = {min(chatter_test_ntokens)}, "
                     f"95%tile = {np.quantile(chatter_test_ntokens, 0.95):.1f}")
        logging.info(f"ANONYMIZED CHATTER dev: max = {max(anonymized_chatter_dev_ntokens)}, "
                     f"min = {min(anonymized_chatter_dev_ntokens)}, "
                     f"95%tile = {np.quantile(anonymized_chatter_dev_ntokens, 0.95):.1f}")
        logging.info(f"ANONYMIZED CHATTER test: max = {max(anonymized_chatter_test_ntokens)}, "
                     f"min = {min(anonymized_chatter_test_ntokens)}, "
                     f"95%tile = {np.quantile(anonymized_chatter_test_ntokens, 0.95):.1f}")
        logging.info(f"PERSONET dev: max = {max(personet_dev_ntokens)}, min = {min(personet_dev_ntokens)}, "
                     f"95%tile = {np.quantile(personet_dev_ntokens, 0.95):.1f}")
        logging.info(f"PERSONET test: max = {max(personet_test_ntokens)}, min = {min(personet_test_ntokens)}, "
                     f"95%tile = {np.quantile(personet_test_ntokens, 0.95):.1f}")
        logging.info("\n\n")

    # create SFT config or Training Args
    kwargs = dict(output_dir=experiments_dir,
                  eval_strategy="steps" if FLAGS.eval else "no",
                  eval_steps=FLAGS.eval_steps,
                  eval_delay=FLAGS.eval_delay,
                  eval_accumulation_steps=FLAGS.eval_accumulation_steps,
                  per_device_train_batch_size=FLAGS.train_batch_size,
                  per_device_eval_batch_size=FLAGS.eval_batch_size,
                  learning_rate=FLAGS.lr,
                  warmup_steps=FLAGS.warmup_steps,
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
                  gradient_checkpointing_kwargs={"use_reentrant": False},
                  save_strategy="no")
    if FLAGS.instrtune:
        config = SFTConfig(max_seq_length=train_instr_seqlen,
                           packing=False,
                           **kwargs)
    else:
        config = TrainingArguments(**kwargs)

    # create compute metrics instance
    compute_metrics = eval_utils.ComputeMetrics(tokenizer)
    if FLAGS.train_dataset == "chatter":
        compute_metrics.eval_df = chatter_dev_df
        compute_metrics.set_dataset("chatter")
    else:
        compute_metrics.eval_df = personet_dev_df
        compute_metrics.set_dataset("personet")

    # create trainer
    log("instantiating trainer")
    if FLAGS.instrtune:
        trainer = SFTTrainer(model=model,
                             args=config,
                             data_collator=DataCollatorForCompletionOnlyLM(response_template=data_utils.ANSWER_TEMPLATE,
                                                                           tokenizer=tokenizer),
                             train_dataset=train_dataset,
                             eval_dataset=(chatter_dev_dataset
                                           if FLAGS.train_dataset == "chatter" else personet_dev_dataset),
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
                          eval_dataset=(chatter_dev_dataset
                                        if FLAGS.train_dataset == "chatter" else personet_dev_dataset),
                          processing_class=tokenizer,
                          compute_metrics=compute_metrics.compute_classification_metrics,
                          callbacks=[callback])

    # train
    log("\n\n\ntraining started")
    log("================================================================================================")
    trainer.train()
    log("\n\n\n==========================================================================================")
    log("training done\n\n\n")

    # plot train loss, dev loss, and dev metric
    log("plotting\n\n")
    if PARTIALSTATE.is_local_main_process:
        with jsonlines.open(logs_file) as reader:
            logs = list(reader)
        metric = "eval_accuracy" if FLAGS.train_dataset == "chatter" else "eval_accuracy@1"
        plot_logs(logs, metric, experiments_dir)

    # predict and evaluate
    test_datasets = [chatter_dev_dataset, chatter_test_dataset, anonymized_chatter_dev_dataset,
                     anonymized_chatter_test_dataset, personet_dev_dataset, personet_test_dataset]
    test_dfs = [chatter_dev_df, chatter_test_df, anonymized_chatter_dev_df, anonymized_chatter_test_df,
                personet_dev_df, personet_test_df]
    test_names = ["CHATTER dev", "CHATTER test", "ANONYMIZED CHATTER dev", "ANONYMIZED CHATTER test",
                  "PERSONET dev", "PERSONET test"]
    for test_dataset, test_df, test_name in zip(test_datasets, test_dfs, test_names):
        log(f"evaluating {test_name}")
        if "CHATTER" in test_name:
            compute_metrics.set_dataset("chatter")
        else:
            compute_metrics.set_dataset("personet")
        compute_metrics.eval_df = test_df
        test_output = trainer.predict(test_dataset)
        log(f"{test_output.metrics}\n\n")
        test_filename = test_name.replace(" ", "_")
        test_predictions_file = os.path.join(experiments_dir, test_filename + ".csv")
        test_metrics_file = os.path.join(experiments_dir, test_filename + ".json")
        if PARTIALSTATE.is_local_main_process:
            test_df.to_csv(test_predictions_file, index=False)
            json.dump(test_output.metrics, open(test_metrics_file, "w"))

    PARTIALSTATE.wait_for_everyone()
    # save model
    if FLAGS.save_model:
        log("saving model")
        trainer.save_model(experiments_dir)

if __name__ == '__main__':
    app.run(finetune)