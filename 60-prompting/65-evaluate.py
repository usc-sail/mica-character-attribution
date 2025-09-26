"""Evaluation functions"""
import data_utils

from absl import app
from absl import flags
import itertools
import numpy as np
import os
import pandas as pd
import random
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

flags.DEFINE_multi_string("evaluation_dir", default=None, help="evaluation directories, give path relative to datadir")
flags.DEFINE_enum("task", default="evaluate", enum_values=["evaluate", "permutation_test"], help="task to perform")
flags.DEFINE_integer("n_tests", default=1, help="number of tests for each pair of evaluation directory")
flags.DEFINE_enum("metric", default="accuracy", enum_values=["accuracy", "precision", "recall", "f1"],
                  help="metric on which to perform test")

def checker(args):
    n_evaluation_dirs = len(args["evaluation_dir"])
    is_test = args["task"] == "permutation_test"
    return n_evaluation_dirs > 0 and (not is_test or n_evaluation_dirs > 1)

flags.register_multi_flags_validator(["task", "evaluation_dir"], checker, ("Provide at least one evaluation directory. "
                                                                           "More than one if doing permutation test"))

FLAGS = flags.FLAGS

def read_evaluation_file(evaluation_file, verbose=True):
    df = pd.read_csv(evaluation_file, index_col=None, dtype={"imdb-id": str})
    df["answer"] = df["response"].str.extract(r"(\w+)")[0].str.lower()
    df["pred"] = df["answer"].apply(lambda ans: 1 if ans == "yes" else 0 if ans == "no" else np.nan)
    n = df["pred"].isna().sum()
    df = df.dropna(subset="pred")
    df = df.groupby(["character", "trope"]).agg({"label": lambda arr: int(arr.values[0]),
                                                 "pred": lambda arr: int(np.any(arr.values == 1))})
    if verbose:
        print(f"could not parse {n} samples")
    return df

def evaluate_runs(evaluation_dir, verbose=True):
    """Evaluate the generator's response"""
    acc_arr, prec_arr, rec_arr, f1_arr = [], [], [], []
    for filename in os.listdir(evaluation_dir):
        if filename.endswith(".csv"):
            if verbose:
                print(filename)
            output_file = os.path.join(evaluation_dir, filename)
            output_df = read_evaluation_file(output_file, verbose=verbose)
            true, pred = output_df["label"], output_df["pred"]
            acc = accuracy_score(true, pred)
            prec, rec, f1, _ = precision_recall_fscore_support(true, pred, average="binary")
            acc_arr.append(acc)
            prec_arr.append(prec)
            rec_arr.append(rec)
            f1_arr.append(f1)
            if verbose:
                print(filename)
                print(f"n={len(true)} samples evaluated")
                print(f"acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}\n")
    acc_mean, acc_std = np.mean(acc_arr), np.std(acc_arr)
    prec_mean, prec_std = np.mean(prec_arr), np.std(prec_arr)
    rec_mean, rec_std = np.mean(rec_arr), np.std(rec_arr)
    f1_mean, f1_std = np.mean(f1_arr), np.std(f1_arr)
    if verbose:
        print(f"acc={acc_mean:.3f} (std={acc_std:.4f})")
        print(f"precision={prec_mean:.3f} (std={prec_std:.4f})")
        print(f"recall={rec_mean:.3f} (std={rec_std:.4f})")
        print(f"F1={f1_mean:.3f} (std={f1_std:.4f})")
    return acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std, len(acc_arr)

def test_statistic(x, y, true, metric):
    if metric == "accuracy":
        val1 = accuracy_score(true, x)
    elif metric == "precision":
        val1 = precision_recall_fscore_support(true, x, average="binary")[0]
    elif metric == "recall":
        val1 = precision_recall_fscore_support(true, x, average="binary")[1]
    else:
        val1 = precision_recall_fscore_support(true, x, average="binary")[2]
    if metric == "accuracy":
        val2 = accuracy_score(true, y)
    elif metric == "precision":
        val2 = precision_recall_fscore_support(true, y, average="binary")[0]
    elif metric == "recall":
        val2 = precision_recall_fscore_support(true, y, average="binary")[1]
    else:
        val2 = precision_recall_fscore_support(true, y, average="binary")[2]
    return val1 - val2

def permutation_test(evaluation_dir1, evaluation_dir2, n_tests=1, metric="accuracy"):
    """Perform permutation between `n_test` random pairs of runs in evaluation_dir1 and evaluation_dir2"""
    filenames1, filenames2 = [], []
    dfs1, dfs2 = [], []
    for filename in os.listdir(evaluation_dir1):
        if filename.endswith(".csv"):
            print(filename)
            output_file = os.path.join(evaluation_dir1, filename)
            df = read_evaluation_file(output_file)
            dfs1.append(df)
            filenames1.append(filename)
    for filename in os.listdir(evaluation_dir2):
        if filename.endswith(".csv"):
            print(filename)
            output_file = os.path.join(evaluation_dir2, filename)
            df = read_evaluation_file(output_file)
            dfs2.append(df)
            filenames2.append(filename)
    pairings = [(i, j) for i in range(len(filenames1)) for j in range(len(filenames2))]
    n = min(len(pairings), n_tests)
    test_pairings = random.sample(pairings, n)
    print(f"X = {metric}({evaluation_dir1})")
    print(f"Y = {metric}({evaluation_dir2})")
    print(f"H0: X = Y vs Ha: X != Y")
    pvalues = []
    for i, (j, k) in enumerate(test_pairings):
        df1, df2 = dfs1[j], dfs2[k]
        df = df1.merge(df2, how="inner", on=("character", "trope"))
        true = df["label_x"].values
        pred1 = df["pred_x"].values
        pred2 = df["pred_y"].values
        test_result = stats.permutation_test((pred1, pred2), statistic=lambda x, y: test_statistic(x, y, true, metric),
                                             permutation_type="samples", n_resamples=1000, alternative="two-sided")
        print(f"test{i + 1}: {filenames1[j]} <-> {filenames2[k]}, {len(df)} samples")
        print(f"test-statistic = X - Y = {test_result.statistic:.6f}, p = {test_result.pvalue:.6f}\n")
        pvalues.append(test_result.pvalue)
    return pvalues

def permutation_test_runs(evaluation_dirs, n_tests=1, metric="accuracy"):
    evaluation_dirs = list(evaluation_dirs)
    n = len(evaluation_dirs) * (len(evaluation_dirs) - 1)
    for evaluation_dir1, evaluation_dir2 in itertools.combinations(evaluation_dirs, r=2):
        pvalues = np.array(permutation_test(evaluation_dir1, evaluation_dir2, n_tests=n_tests, metric=metric))
        corrected_pvalues = n * pvalues
        print(f"corrected p = {corrected_pvalues}")
        print()

def main(_):
    evaluation_dirs = [os.path.join(data_utils.DATADIR, dr) for dr in FLAGS.evaluation_dir]
    if FLAGS.task == "evaluate":
        evaluation_dir = evaluation_dirs[0]
        evaluate_runs(evaluation_dir)
    else:
        permutation_test_runs(evaluation_dirs, FLAGS.n_tests, FLAGS.metric)

if __name__ == '__main__':
    app.run(main)