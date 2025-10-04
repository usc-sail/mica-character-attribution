"""Evaluation class for instruction-tuning and classification models"""
import numpy as np
import pandas as pd
import re
import scipy.special
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, f1_score
from transformers.trainer import EvalPrediction
from typing import Dict

class ComputeMetrics:
    """Compute metrics class for classification and instruction-tuning methods for CHATTER & PERSONET datasets"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eval_df = None
        self.dataset = None

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute metrics for different datasets"""
        if self.dataset == "chatter-contexts" or self.dataset == "chatter-segments":
            true_arr, pred_arr = [], []
            for _, sample_df in self.eval_df.groupby("key"):
                nonnan_sample_df = sample_df.dropna(subset="pred")
                if len(nonnan_sample_df) > 0:
                    true = nonnan_sample_df["label"].values[0]
                    if any(nonnan_sample_df["pred"] == 1):
                        pred = 1
                    else:
                        pred = 0
                    true_arr.append(true)
                    pred_arr.append(pred)
            if len(true_arr) > 0:
                acc = accuracy_score(true_arr, pred_arr)
                prec, rec, f1, _ = precision_recall_fscore_support(true_arr, pred_arr, average="binary")
            else:
                acc, prec, rec, f1 = np.nan, np.nan, np.nan, np.nan
            metrics = {"accuracy": acc,
                       "precision": prec,
                       "recall": rec,
                       "f1": f1,
                       "total_nsamples": len(self.eval_df["key"].unique()),
                       "eval_nsamples": len(true_arr)}
        else:
            n_correct = (self.eval_df["label"] == self.eval_df["pred"]).sum()
            n_eval = self.eval_df["pred"].notna().sum()
            n_total = len(self.eval_df)
            acc = n_correct/n_eval
            metrics = {"accuracy": acc, "total_nsamples": n_total, "eval_nsamples": n_eval}
        return metrics

    def compute_sft_metrics(self, evalprediction: EvalPrediction) -> Dict[str, float]:
        """Compute metrics for the supervized fine-tuning method"""

        # function to codify label text
        def convert_label_text_to_label(row):
            if row["label_text == "yes":
                return 1
            elif label_text == "no":
                return 0
            elif label_text in ["1", "2", "3", "4", "5"]:
                return int(label_text)
            else:
                return np.nan

        label_ids = evalprediction.label_ids
        prediction_ids = evalprediction.predictions
        prediction_texts = []
        for sample_label_ids, sample_prediction_ids in zip(label_ids, prediction_ids):
            i = np.nonzero(sample_label_ids != -100)[0]
            sample_completion_ids = sample_prediction_ids[i - 1:]
            sample_prediction_text = self.tokenizer.decode(sample_completion_ids, skip_special_tokens=True).lower()
            sample_prediction_text = re.sub("\s+", "", sample_prediction_text)
            prediction_texts.append(sample_prediction_text)
        self.eval_df["pred-text"] = prediction_texts
        self.eval_df["pred"] = self.eval_df.apply(
        return self._compute_metrics()

    def compute_chr_metrics(self, evalprediction: EvalPrediction) -> Dict[str, float]:
        """Compute metrics for the character representations model"""
        logits = evalprediction.predictions
        probs = scipy.special.softmax(logits, axis=-1)
        if self.multiclass:
            self.eval_df["pred"] = np.argmax(probs, axis=-1)
            self.eval_df = pd.concat((self.eval_df,
                                      pd.DataFrame(probs, columns=[f"prob{i + 1}" for i in range(probs.shape[1])])),
                                      axis=1)
        else:
            self.eval_df["pred"] = np.argmax(probs, axis=-1)
            self.eval_df["prob"] = np.max(probs, axis=-1)
        return self._compute_metrics()