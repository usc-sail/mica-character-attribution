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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer import EvalPrediction
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

flags.DEFINE_string("data_file", default=None, help="jsonlines file of samples")
flags.DEFINE_string("train_file", default=None, help="jsonlines file of training samples")
flags.DEFINE_string("dev_file", default=None, help="jsonlines file of dev samples")
flags.DEFINE_string("test_file", default=None, help="jsonlines file of test samples")
flags.DEFINE_string("model", default="meta-llama/Llama-3.1-8B-Instruct", help="huggingface model name")
flags.DEFINE_bool("bf16", default=False, help="use brain floating point (default=fp16)")
flags.DEFINE_bool("load_4bit", default=False, help="do 4-bit QLoRA training")
flags.DEFINE_bool("load_8bit", default=False, help="do 8-bit QLoRA training")
flags.DEFINE_bool("flash_attn", default=False, help="use flash attention (default=scaled dot product 'sdpa')")
flags.DEFINE_integer("train_batch_size", default=1, help="training batch size")
flags.DEFINE_integer("eval_batch_size", default=1, help="evaluation batch size")
flags.DEFINE_float("lr", default=1e-5, help="learning rate")
flags.DEFINE_string("optim", default="adamw_torch", help=("optimizer name (https://github.com/huggingface/"
                                                          "transformers/blob/main/src/transformers/training_args.py)"))
flags.DEFINE_integer("max_seq_len", default=1024, help="maximum sequence length")
flags.DEFINE_integer("train_steps", default=1024, help="training steps")
flags.DEFINE_integer("eval_steps", default=64, help="training steps between successive evaluations")
flags.DEFINE_integer("logging_steps", default=8, help="training steps between successive logging")
flags.DEFINE_bool("eval_testset", default=False, help="evaluate test set every eval_steps (default=dev set)")
flags.DEFINE_multi_string("lora_target_module", default=["q_proj", "k_proj"], help="target modules to train using LoRA")
flags.DEFINE_multi_string("lora_save_module", default=["lm_head"], help="modules to train normally")
flags.DEFINE_integer("rank", default=8, help="lora rank")
flags.DEFINE_integer("alpha", default=16, help="lora alpha")
flags.DEFINE_float("dropout", default=0, help="dropout")
flags.DEFINE_float("weight_decay", default=0, help="weight decay")
flags.DEFINE_float("max_grad_norm", default=10, help="maximum gradient norm")

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
            "\n\nCHARACTER: $CHARACTER$\n\nSTORY: $STORY$\n\n ANSWER: $ANSWER$")
ANSWER_TEMPLATE = " ANSWER:"

def formatter(examples):
    texts = []
    for i in range(len(examples)):
        text = (TEMPLATE
                .replace("$ANSWER$", "yes" if examples["label"][i] == 1 else "no")
                .replace("$CHARACTER$", examples["name"][i])
                .replace("$ATTRIBUTE$", examples["definition"][i])
                .replace("$STORY$", examples["text"][i])
                )
        texts.append(text)
    return texts

def compute_metrics(tokenizer, evalprediction: EvalPrediction):
    labels = evalprediction.label_ids
    predictions = evalprediction.predictions.argmax(axis=-1)
    rx, cx = np.where(labels != -100)
    labels = list(map(lambda x: x.strip().lower(), tokenizer.batch_decode(labels[rx, cx].reshape(-1, 1))))
    labels = list(map(lambda x: 1 if x == "yes" else 0, labels))
    labels = np.array(labels)
    predictions = list(map(lambda x: x.strip().lower(), tokenizer.batch_decode(predictions[rx, cx - 1].reshape(-1, 1))))
    predictions = list(map(lambda x: 1 if x == "yes" else 0 if x == "no" else np.nan, predictions))
    predictions = np.array(predictions)
    labels = labels[~np.isnan(predictions)]
    predictions = predictions[~np.isnan(predictions)]
    if predictions:
        acc = accuracy_score(labels, predictions)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    else:
        acc, prec, rec, f1 = 0, 0, 0, 0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def finetune(_):
    """Instruction Fine-tune LLMs on the character attribution classification task"""
    # read data
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

    # print template
    logging.info(f"template:\n{TEMPLATE}")

    # instantiate model
    logging.info("instantiating model")
    partial_state = PartialState()
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
    logging.info("instantiating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # create datasets
    logging.info("creating datasets")
    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)
    test_dataset = Dataset.from_list(test_data)

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
    experiments_dir = os.path.join(datadirs.datadir, "50-modeling/finetune")
    config = SFTConfig(output_dir=experiments_dir,
                       eval_strategy="steps",
                       eval_steps=FLAGS.eval_steps,
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
                       packing=False)

    # create trainer
    logging.info("instantiating trainer")
    trainer = SFTTrainer(model=model,
                         args=config,
                         data_collator=DataCollatorForCompletionOnlyLM(response_template=ANSWER_TEMPLATE,
                                                                       tokenizer=tokenizer),
                         train_dataset=train_dataset,
                         eval_dataset=test_dataset if FLAGS.eval_testset else dev_dataset,
                         peft_config=lora_config,
                         formatting_func=formatter,
                         compute_metrics=lambda x: compute_metrics(tokenizer, x))

    # train
    logging.info("training")
    trainer.train()

if __name__ == '__main__':
    app.run(finetune)