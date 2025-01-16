"""Train a classification model"""
import datadirs

from absl import app
from absl import flags
from accelerate import PartialState
from accelerate.logging import get_logger
from datasets import Dataset
from logging import FileHandler, StreamHandler, Formatter
import os
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import scipy.special
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, Trainer, TrainingArguments

FLAGS = flags.FLAGS

# input and model
flags.DEFINE_enum("source", default="all", enum_values=["all", "utter", "desc"],
                  help="source of character information from story")
flags.DEFINE_enum("context", default="0", enum_values=["0", "1", "5", "10", "20"],
                  help="number of neighboring segments")
flags.DEFINE_integer("loglen", default=13, lower_bound=13, upper_bound=15, help="sequence length")

# hyper parameters
flags.DEFINE_integer("eval_steps", default=8, help="number of update steps between two evaluations")
flags.DEFINE_integer("train_batch_size", default=4, help="train batch size")
flags.DEFINE_integer("eval_batch_size", default=8, help="eval batch size")
flags.DEFINE_float("lr", default=1e-4, help="learning rate")
flags.DEFINE_string("optim", default="adamw_torch", help="optimizer name")
flags.DEFINE_float("weight_decay", default=0, help="weight decay")
flags.DEFINE_float("max_grad_norm", default=1, help="maximum gradient norm")
flags.DEFINE_integer("max_steps", default=64, help="total number of training steps")
flags.DEFINE_integer("rank", default=8, help="lora rank")
flags.DEFINE_integer("alpha", default=16, help="lora alpha")
flags.DEFINE_float("dropout", default=0, help="dropout")

def compute_loss(inputs, labels, **kwargs):
    return torch.nn.functional.cross_entropy(inputs.logits, labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = scipy.special.softmax(logits, axis=-1)
    preds = probs.argmax(axis=-1)
    m_pos = (preds == 1).sum()
    m_neg = (preds == 0).sum()
    f_pos = m_pos/(m_pos + m_neg + 1e-23)
    f_neg = m_neg/(m_pos + m_neg + 1e-23)
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    acc_pos = ((preds == 1) & (labels == 1)).sum()/n_pos
    acc_neg = ((preds == 0) & (labels == 0)).sum()/n_neg
    acc = accuracy_score(labels, preds)
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, preds, average="binary")
    auc = roc_auc_score(labels, probs[:, 1])
    return {"frac-pos": f_pos, "frac-neg": f_neg, "acc-pos": acc_pos, "acc-neg": acc_neg, "acc": acc, "auc": auc,
            "precision": precision, "recall": recall, "f1": fscore}

def train(_):
    # get K-length
    Ksize = 1 << (FLAGS.loglen - 10)

    # create experiments directory
    experiments_dir = os.path.join(
        datadirs.datadir,
        f"70-classification/results/{FLAGS.source}-context-{FLAGS.context}-llama3-8B-{Ksize}K")
    os.makedirs(experiments_dir, exist_ok=True)

    partial_state = PartialState()

    # creater training arguments
    trainer_arguments = TrainingArguments(output_dir=experiments_dir,
                                          eval_strategy="steps",
                                          eval_steps=FLAGS.eval_steps,
                                          per_device_train_batch_size=FLAGS.train_batch_size,
                                          per_device_eval_batch_size=FLAGS.eval_batch_size,
                                          learning_rate=FLAGS.lr,
                                          weight_decay=FLAGS.weight_decay,
                                          max_grad_norm=FLAGS.max_grad_norm,
                                          logging_strategy="steps",
                                          logging_steps=4,
                                          log_level="info",
                                          save_strategy="steps",
                                          save_steps=FLAGS.eval_steps,
                                          save_total_limit=2,
                                          load_best_model_at_end=True,
                                          metric_for_best_model="f1",
                                          save_only_model=True,
                                          seed=2025,
                                          data_seed=2025,
                                          optim=FLAGS.optim,
                                          max_steps=FLAGS.max_steps,
                                          bf16=True,
                                          report_to="tensorboard",
                                          ddp_find_unused_parameters=False,
                                          )

    # get logger
    os.makedirs(trainer_arguments.logging_dir, exist_ok=True)
    logger = get_logger(__name__)
    handlers = [FileHandler(os.path.join(trainer_arguments.logging_dir, "run.log"))]
    for handler in handlers:
        handler.setFormatter(None)
        logger.logger.addHandler(handler)

    # log command-line arguments
    for module in FLAGS.flags_by_module_dict():
        if module.endswith("74-train-ddp.py"):
            for flagobj in FLAGS.flags_by_module_dict()[module]:
                logger.info(f"{flagobj.name:20s} {flagobj.help:70s} = {flagobj.value}", main_process_only=True)

    # get model
    logger.info("loading model")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16,
                                             bnb_4bit_use_double_quant=True,
                                             bnb_4bit_quant_storage=torch.bfloat16)
    model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.1-8B",
                                                               num_labels=2,
                                                               quantization_config=quantization_config,
                                                               device_map={"": partial_state.process_index},
                                                               torch_dtype=torch.bfloat16)
    model.config.pad_token_id = model.config.eos_token_id
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                             target_modules=["q_proj", "v_proj"],
                             r=FLAGS.rank,
                             lora_alpha=FLAGS.alpha,
                             lora_dropout=FLAGS.dropout,
                             use_rslora=True,
                             bias="none")
    model = get_peft_model(model, lora_config)
    n_trainable, n_all = model.get_nb_trainable_parameters()
    logger.info(f"{n_trainable} trainable params, {n_all} all params", main_process_only=True)

    # get data
    logger.info("loading data")
    tensors_dir = os.path.join(datadirs.datadir,
                               f"70-classification/data/{FLAGS.source}-context-{FLAGS.context}-llama3-{Ksize}K")
    tokenids_file = os.path.join(tensors_dir, "tokens.pt")
    labels_file = os.path.join(tensors_dir, "labels.csv")
    tokenids = torch.load(tokenids_file, weights_only=True)
    labels_df = pd.read_csv(labels_file, index_col=None)

    # divide into train, dev, and test
    partition = labels_df["partition"]
    labels = torch.tensor(labels_df["label"], dtype=int)
    trn_tokenids, trn_labels = tokenids[partition == "train"], labels[partition == "train"]
    dev_tokenids, dev_labels = tokenids[partition == "dev"], labels[partition == "dev"]
    tst_tokenids, tst_labels = tokenids[partition == "test"], labels[partition == "test"]
    trn_dataset = Dataset.from_dict({"input_ids": trn_tokenids, "labels": trn_labels})
    dev_dataset = Dataset.from_dict({"input_ids": dev_tokenids, "labels": dev_labels})
    tst_dataset = Dataset.from_dict({"input_ids": tst_tokenids, "labels": tst_labels})
    
    # create trainer
    trainer = Trainer(model=model, args=trainer_arguments, train_dataset=trn_dataset, eval_dataset=tst_dataset,
                      compute_metrics=compute_metrics, compute_loss_func=compute_loss)

    # train the model
    trainer.train()

    # evaluate best model on test set
    perf = trainer.evaluate(tst_dataset)
    logger.info(f"test performance: {perf}", main_process_only=True)

if __name__ == '__main__':
    app.run(train)