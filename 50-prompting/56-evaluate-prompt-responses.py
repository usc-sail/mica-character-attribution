"""Evaluate the responses obtained from prompting"""
import pandas as pd
import numpy as np
import re
import os

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string("datadir", default=None, required=True, help="data directory")
flags.DEFINE_string("promptresponsefile", default=None, required=True, help="csv file containing prompt responses")
flags.DEFINE_string("testdatafile", default="60-modeling/test-dataset.csv", help="test dataset file")

def text_to_label(text):
    if pd.notna(text):
        text = re.sub(r"^\W+", "", text).lower()
        if text.startswith("no"):
            return 0
        elif text.startswith("yes"):
            return 1
        else:
            return np.nan
    else:
        return np.nan

def agg_preds(preds):
    nonnan_preds = preds[~np.isnan(preds)]
    if len(nonnan_preds) > 0:
        return int(nonnan_preds.sum() > 0)
    else:
        return np.nan

def evaluate_prompt_responses(_):
    data_dir = FLAGS.datadir
    prompt_response_file = os.path.join(data_dir, FLAGS.promptresponsefile)
    test_file = os.path.join(data_dir, FLAGS.testdatafile)

    response_df = pd.read_csv(prompt_response_file, index_col=None)
    test_df = pd.read_csv(test_file, index_col=None)

    ncandidates = sum([column.startswith("response") for column in response_df.columns])
    for i in range(ncandidates):
        response_df[f"pred-{i + 1}"] = response_df[f"response-{i + 1}"].apply(text_to_label)

    pred_columns = [f"pred-{i + 1}" for i in range(ncandidates)]
    pred_column = []
    for _, row in response_df.iterrows():
        preds = np.array([row[pred_column] for pred_column in pred_columns])
        preds = preds[~np.isnan(preds)]
        if len(preds) > 0 and preds.mean() != 0.5:
            pred = int(preds.mean() > 0)
        else:
            pred = np.nan
        pred_column.append(pred)
    response_df["pred"] = pred_column

    pred_df = response_df.groupby(["character", "trope"])["pred"].agg(agg_preds).reset_index()
    pred_df = pred_df.merge(test_df[["character", "trope", "conf"]], how="inner", on=["character", "trope"])
    assert len(pred_df) == len(test_df)
    n = len(pred_df)

    pred_df["true-polarity"] = (pred_df["conf"] > 0).astype(int)
    pred_df["true-weight"] = (pred_df["conf"].abs() - 2)/4

    pred_df = pred_df[pred_df["pred"].notna()]
    m = len(pred_df)
    percentage_missed = 100 * (n - m)/n

    true = pred_df["true-polarity"].values
    weight = pred_df["true-weight"].values
    pred = pred_df["pred"].values

    tparr = (true == 1) & (pred == 1)
    fparr = (true == 0) & (pred == 1)
    fnarr = (true == 1) & (pred == 0)
    tp = tparr.sum()
    wtp = (weight * tparr).sum()
    fp = fparr.sum()
    wfp = (weight * fparr).sum()
    fn = fnarr.sum()
    wfn = (weight * fnarr).sum()
    precision = tp/(tp + fp)
    wprecision = wtp/(wtp + wfp)
    recall = tp/(tp + fn)
    wrecall = wtp/(wtp + wfn)
    f1 = 2 * precision * recall / (precision + recall)
    wf1 = 2 * wprecision * wrecall / (wprecision + wrecall)

    print(f"{m}/{n} evaluated ({percentage_missed:.1f}% missed)")
    print(f"unweighted: precision = {precision:.3f} recall = {recall:.3f} f1 = {f1:.3f}")
    print(f"weighted:   precision = {wprecision:.3f} recall = {wrecall:.3f} f1 = {wf1:.3f}")

if __name__ == '__main__':
    app.run(evaluate_prompt_responses)