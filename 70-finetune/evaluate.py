"""Evaluation class for instruction-tuning and classification models"""
import data

import numpy as np
import pandas as pd
import re
import scipy.special
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from transformers.trainer import EvalPrediction
from typing import Dict

class ComputeMetrics:
    """
    Compute metrics class for classification and instruction-tuning methods for 
    CHATTER & PERSONET datasets
    """

    def __call__(
            self,
            dataset_name: str,
            eval_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute metrics for different datasets
        """
        if (dataset_name == "chatter-contexts"
                or dataset_name == "chatter-segments"):
            true_arr, pred_arr = [], []
            for _, sample_df in eval_df.groupby("key"):
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
                prec, rec, f1, _ = precision_recall_fscore_support(
                    true_arr, pred_arr, average="binary")
            else:
                acc, prec, rec, f1 = np.nan, np.nan, np.nan, np.nan
            metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "total_nsamples": len(eval_df["key"].unique()),
                "eval_nsamples": len(true_arr)}
        else:
            n_correct = (eval_df["label"] == eval_df["pred"]).sum()
            n_eval = int(eval_df["pred"].notna().sum())
            n_total = len(eval_df)
            acc = n_correct/n_eval
            metrics = {
                "accuracy": acc,
                "total_nsamples": n_total,
                "eval_nsamples": n_eval}
        return metrics