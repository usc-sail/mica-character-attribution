"""Evaluation class for instruction-tuning and classification models"""
import numpy as np
import pandas as pd
import scipy.special
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, f1_score
from transformers.trainer import EvalPrediction
from typing import Dict

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
            self.eval_df.sort_values(by=["key", "attribute"], inplace=True)
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
        predictions = evalprediction.predictions
        probs = predictions[:, :, 0]
        output_ids = predictions[:, :, 1].astype(int)
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
        probs = scipy.special.softmax(logits, axis=-1)
        self.eval_df["pred"] = np.argmax(probs, axis=-1)
        self.eval_df["prob"] = np.max(probs, axis=-1)
        return self._compute_metrics()