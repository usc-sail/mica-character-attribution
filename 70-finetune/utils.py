"""Utility functions for training, evaluation, and prediction"""

from absl import logging
import jsonlines
import matplotlib.pyplot as plt
import os
import pandas as pd
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import List, Dict

# define logging callback class
class LoggerCallback(TrainerCallback):
    """Callback class to log to file and handle general logging"""
    def __init__(self, experiments_dir):
        super().__init__()
        logging.get_absl_handler().use_absl_log_file(program_name="train", log_dir=experiments_dir)
        logging.get_absl_handler().setFormatter(None)
        self.logs_file = os.path.join(experiments_dir, "logs.jsonl")
        self.logs_writer = jsonlines.open(self.logs_file, mode="w", flush=True)

    def on_log(self,
               args: TrainingArguments,
               state: TrainerState,
               control: TrainerControl,
               logs: Dict[str, float],
               **kwargs):
        logging.info(f"STEP {state.global_step}/{state.max_steps}")
        for logkey, logvalue in logs.items():
            logging.info(f"{logkey} = {logvalue:.6f}")
        logging.info("\n")
        logs["step"] = state.global_step
        self.logs_writer.write(logs)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.logs_writer.close()

def plot_logs(logs: List[Dict[str, float]], metric: str, experiments_dir: str):
    """plot the train loss, dev loss and dev evaluation metric, and also save a dataframe file containing the same 
    information
    """
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