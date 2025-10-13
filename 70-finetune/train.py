"""Train LLMs on the character attribution task"""
import crm
import data
import sft

from absl import app
from absl import flags
from absl import logging
from accelerate import PartialState
from datetime import datetime
import numpy as np
import os
import random
import torch

# training corpus
flags.DEFINE_enum(
    "train_dataset",
    default="chatter-contexts",
    enum_values=["chatter-contexts", "personet"],
    help="training dataset")

# CHATTER arguments
flags.DEFINE_enum(
    "chatter_truncation_strategy",
    default="first",
    enum_values=["first", "semantic"],
    help=("truncation strategy on character segments to create the chatter "
          "contexts documents"))
flags.DEFINE_enum(
    "chatter_size",
    default="2000",
    enum_values=["250", "500", "1000", "1500", "2000"],
    help="size of the chatter contexts documents in words")

# dataset processing arguments
flags.DEFINE_integer(
    "tokenization_batch_size",
    default=4096,
    help="batch size for tokenization")

# model arguments
flags.DEFINE_enum(
    "model",
    default="sft",
    enum_values=["sft", "crm"],
    help=("do supervized fine-tuning or character representation modeling"))
flags.DEFINE_string(
    "modelname",
    default="meta-llama/Llama-3.1-8B-Instruct",
    help="huggingface model name")

# peft arguments
flags.DEFINE_bool(
    "bf16",
    default=False,
    help="use brain floating point (default=fp16)")
flags.DEFINE_bool("load_4bit", default=False, help="do 4-bit QLoRA training")
flags.DEFINE_bool("load_8bit", default=False, help="do 8-bit QLoRA training")
flags.DEFINE_enum(
    "attn",
    default="sdpa",
    enum_values=["flash_attention_2", "sdpa", "eager"],
    help="attention implementation")
flags.DEFINE_multi_string(
    "lora_target_module",
    default=["q_proj", "k_proj"],
    help="target modules to train using LoRA")
flags.DEFINE_integer("rank", default=32, help="lora rank")
flags.DEFINE_integer("alpha", default=64, help="lora alpha")
flags.DEFINE_float("dropout", default=0, help="dropout")

# training arguments
flags.DEFINE_integer(
    "train_batch_size",
    default=1,
    help="per GPU training batch size")
flags.DEFINE_integer(
    "gradient_accumulation_steps",
    default=1,
    help="number of gradient accumulation steps")
flags.DEFINE_bool(
    "gradient_checkpointing",
    default=False,
    help="enable gradient checkpointing")
flags.DEFINE_float("lr", default=2e-5, help="learning rate")
flags.DEFINE_integer(
    "warmup_steps",
    default=0,
    help="number of steps used for linear warmup from 0 to lr")
flags.DEFINE_string(
    "optim",
    default="adamw_torch",
    help=("optimizer name "
          "(https://github.com/huggingface/transformers/blob/main/src/transformers/"
          "training_args.py)"))
flags.DEFINE_integer("train_steps", default=1024, help="training steps")
flags.DEFINE_float("weight_decay", default=0, help="weight decay")
flags.DEFINE_float("max_grad_norm", default=10, help="maximum gradient norm")

# evaluation arguments
flags.DEFINE_bool("eval", default=True, help="do evaluation during training")
flags.DEFINE_integer(
    "eval_steps",
    default=64,
    help="training steps between successive evaluations")
flags.DEFINE_integer("eval_batch_size", default=1, help="evaluation batch size")
flags.DEFINE_integer(
    "eval_accumulation_steps",
    default=None,
    help=("number of prediction steps to accumulate the output tensors for, "
          "before moving to CPU"))
flags.DEFINE_bool(
    "eval_on_start",
    default=False,
    help="Evaluate dev set at the start of training")
flags.DEFINE_integer(
    "eval_delay",
    default=None,
    help=("Number of epochs or steps to wait for before the first evaluation "
          "can be performed"))

# logging arguments
flags.DEFINE_integer(
    "logging_steps",
    default=8,
    help="training steps between successive logging")

# save model arguments
flags.DEFINE_bool(
    "save_model",
    default=False,
    help="save model safetensors at the end of training")

FLAGS = flags.FLAGS
PARTIALSTATE = PartialState()
torch.cuda.set_device(PARTIALSTATE.local_process_index)

# set seeds
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

def log(message):
    if PARTIALSTATE.is_local_main_process:
        logging.info(message)

def get_experiments_directory():
    # decide experiments directory
    model_dir = FLAGS.model
    if FLAGS.train_dataset == "chatter-contexts":
        train_part = f"chatter-{FLAGS.chatter_truncation_strategy}-{FLAGS.chatter_size}"
    else:
        train_part = "personet"
    modelname_part = FLAGS.modelname.split("/")[-1]
    if FLAGS.load_4bit:
        modelname_part += "-4bit"
    elif FLAGS.load_8bit:
        modelname_part += "-8bit"
    if FLAGS.bf16:
        modelname_part += "-bf16"
    else:
        modelname_part += "-fp16"
    datetime_part = datetime.strftime(datetime.now(), "%Y%b%d-%H%M%S")
    filename = f"{train_part}--{modelname_part}--{datetime_part}"
    experiments_dir = os.path.join(
        data.DATADIR, "70-finetune", model_dir, filename)
    return experiments_dir

def log_arguments():
    logging.info("ARGUMENTS")

    # log training dataset
    logging.info("=" * 80)
    logging.info(f"{'train dataset':30s} = {FLAGS.train_dataset}")
    if FLAGS.train_dataset == "chatter-contexts":
        logging.info(
            f"{'chatter truncation strategy':30s} = "
            f"{FLAGS.chatter_truncation_strategy}")
        logging.info(
            f"{'chatter document size':30s} = {FLAGS.chatter_size} words")

    # log model arguments
    logging.info(f"{'model':30s} = {FLAGS.model}")
    logging.info(f"{'modelname':30s} = {FLAGS.modelname}")

    # log peft arguments
    precision = "bf16" if FLAGS.bf16 else "fp16"
    logging.info(f"{'precision':30s} = {precision}")
    quantization = "4bit" if FLAGS.load_4bit else "8bit" if FLAGS.load_8bit else "NONE"
    logging.info(f"{'quantization':30s} = {quantization}")
    logging.info(f"{'attention':30s} = {FLAGS.attn}")
    logging.info(f"{'LoRA target-modules':30s} = {FLAGS.lora_target_module}")
    logging.info(f"{'LoRA rank':30s} = {FLAGS.rank}")
    logging.info(f"{'LoRA alpha':30s} = {FLAGS.alpha}")
    logging.info(f"{'LoRA dropout':30s} = {FLAGS.dropout}")

    # log training arguments
    logging.info(f"{'train-batch-size':30s} = {FLAGS.train_batch_size}")
    logging.info(f"{'learning-rate':30s} = {FLAGS.lr}")
    logging.info(
        f"{'gradient-accumulation':30s} = {FLAGS.gradient_accumulation_steps}")
    logging.info(
        f"{'gradient-checkpointing':30s} = {FLAGS.gradient_checkpointing}")
    logging.info(f"{'warmup-steps':30s} = {FLAGS.warmup_steps}")
    logging.info(f"{'optimizer':30s} = {FLAGS.optim}")
    logging.info(f"{'weight-decay':30s} = {FLAGS.weight_decay}")
    logging.info(f"{'max-grad-norm':30s} = {FLAGS.max_grad_norm}")
    logging.info(f"{'train-steps':30s} = {FLAGS.train_steps}")

    # log evaluation arguments
    logging.info(f"{'do-eval':30s} = {FLAGS.eval}")
    logging.info(f"{'eval-batch-size':30s} = {FLAGS.eval_batch_size}")
    logging.info(
        f"{'eval-accumulation-steps':30s} = {FLAGS.eval_accumulation_steps}")
    logging.info(f"{'eval-on-start':30s} = {FLAGS.eval_on_start}")
    logging.info(f"{'eval-delay':30s} = {FLAGS.eval_delay}")
    logging.info(f"{'eval-steps':30s} = {FLAGS.eval_steps}")

    # log logging arguments
    logging.info(f"{'logging-steps':30s} = {FLAGS.logging_steps}")

    # log model saving arguments
    logging.info(f"{'save-model':30s} = {FLAGS.save_model}")

    logging.info("=" * 80 + "\n")

def train(_):
    """
    Do supervized fine-tuning or model character representations for the 
    character attribution task
    """
    # get and create experiments directory
    experiments_dir = get_experiments_directory()
    if PARTIALSTATE.is_local_main_process:
        os.makedirs(experiments_dir, exist_ok=True)
    PARTIALSTATE.wait_for_everyone()

    # log arguments
    if PARTIALSTATE.is_local_main_process:
        log_arguments()

    # read chatter data
    log("reading chatter data")
    chatter_data = data.load_chatter_contexts(
        truncation_strategy=FLAGS.chatter_truncation_strategy,
        size_in_words=int(FLAGS.chatter_size),
        anonymize=False)
    chatter_train_data = list(
        filter(lambda obj: obj["partition"] == "train", chatter_data))
    chatter_dev_data = list(
        filter(lambda obj: obj["partition"] == "dev", chatter_data))
    chatter_test_data = list(
        filter(lambda obj: obj["partition"] == "test", chatter_data))

    # read personet data
    log("reading personet data")
    personet_data = data.load_personet()
    personet_train_data = list(
        filter(lambda obj: obj["partition"] == "train", personet_data))
    personet_dev_data = list(
        filter(lambda obj: obj["partition"] == "dev", personet_data))
    personet_test_data = list(
        filter(lambda obj: obj["partition"] == "test", personet_data))

    # log data sizes
    if PARTIALSTATE.is_local_main_process:
        logging.info(f"{len(chatter_train_data)} CHATTER train examples")
        logging.info(f"{len(chatter_dev_data)} CHATTER dev examples")
        logging.info(f"{len(chatter_test_data)} CHATTER test examples")
        logging.info(f"{len(personet_train_data)} PERSONET train examples")
        logging.info(f"{len(personet_dev_data)} PERSONET dev examples")
        logging.info(f"{len(personet_test_data)} PERSONET test examples\n")

    # assign train and dev data
    if FLAGS.train_dataset == "chatter-contexts":
        train_data = chatter_train_data
        dev_data = chatter_dev_data
    else:
        train_data = personet_train_data
        dev_data = personet_dev_data

    if FLAGS.model == "sft":
        sft.train(partial_state=PARTIALSTATE,
                  experiments_dir=experiments_dir,
                  train_data=train_data,
                  dev_data=dev_data,
                  train_and_dev_dataset_name=FLAGS.train_dataset)
    else:
        crm.train(partial_state=PARTIALSTATE,
                  experiments_dir=experiments_dir,
                  train_data=train_data,
                  dev_data=dev_data,
                  train_and_dev_dataset_name=FLAGS.train_dataset)

if __name__ == '__main__':
    app.run(train)