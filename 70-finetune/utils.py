"""Utility functions for training, evaluation, and prediction"""

from absl import logging
import gc
import jsonlines
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from transformers import TrainingArguments
from transformers import TrainerCallback
from transformers import TrainerControl
from transformers import TrainerState
from typing import List, Dict

# define logging callback class
class LoggerCallback(TrainerCallback):
    """Callback class to log to file and handle general logging"""
    def __init__(self, experiments_dir):
        super().__init__()
        logging.get_absl_handler().use_absl_log_file(
            program_name="train", log_dir=experiments_dir)
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

    def on_train_end(self,
                     args: TrainingArguments,
                     state: TrainerState,
                     control: TrainerControl,
                     **kwargs):
        self.logs_writer.close()

def plot_logs(logs: List[Dict[str, float]], metric: str, experiments_dir: str):
    """
    Plot the train loss, dev loss and dev evaluation metric, and also save a 
    dataframe file containing the same information
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
    plt.plot(train_steps, train_loss, color="blue", lw=5, marker="o", ms=15,
             label="train loss")
    plt.plot(dev_steps, dev_loss, color="red", lw=5, marker="s", ms=15,
             label="dev loss")
    plt.plot(dev_steps, dev_metric, color="green", lw=5, marker="^", ms=15,
             label=f"dev {metric}")
    plt.ylabel("metric")
    plt.xlabel("step")
    plt.legend(fontsize="x-large")
    plots_file = os.path.join(experiments_dir, "plot.png")
    progress_file = os.path.join(experiments_dir, "progress.csv")
    plt.savefig(plots_file)
    df = pd.DataFrame(rows, columns=["step", "partition", "loss", metric])
    df.to_csv(progress_file, index=False)

def release_training_memory(trainer):
    """
    Clean up GPU and CPU memory used during training, keeping model + DeepSpeed 
    ZeRO partitions intact for inference.
    Safe to call between trainer.train() and trainer.predict().
    """

    # Ensure model exists
    if not hasattr(trainer, "model") or trainer.model is None:
        return

    # Zero out gradients and free them
    try:
        trainer.model.zero_grad(set_to_none=True)
    except Exception:
        pass

    # Remove optimizer state to free ZeRO / Adam shards
    if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
        try:
            # Handle DeepSpeed-wrapped optimizer if present
            opt = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
            if hasattr(opt, "param_groups"):
                for group in opt.param_groups:
                    for p in group.get("params", []):
                        p.grad = None
            opt.param_groups = []
        except Exception:
            pass
        trainer.optimizer = None

    # Remove LR scheduler (not needed after training)
    if hasattr(trainer, "lr_scheduler"):
        trainer.lr_scheduler = None

    # Disable gradient checkpointing (if active)
    if hasattr(trainer.model, "gradient_checkpointing_disable"):
        try:
            trainer.model.gradient_checkpointing_disable()
        except Exception:
            pass

    # Switch to eval mode to disable dropout, grads, etc.
    try:
        trainer.model.eval()
    except Exception:
        pass

    # Garbage collect and clear CUDA memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()