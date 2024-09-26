import os
import numpy as np
import pandas as pd

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_multi_string("i", default=[], help="mturk batch result csv file")
flags.DEFINE_string("o", default=None, help="output csv file")
flags.register_validator("i", lambda value: len(value) > 0, "input cannot be empty")

def calculate_worker_performance(_):
    worker_rows = []
    batch_dfs = []
    for input_file in FLAGS.i:
        batch_df = pd.read_csv(os.path.join(FLAGS.data_dir, input_file), index_col=None)
        batch_dfs.append(batch_df)
    annotations_df = pd.concat(batch_dfs, axis=0)

    for worker_id, worker_df in annotations_df.groupby("WorkerId"):
        n_ratings = len(worker_df)
        n_positive_labels = (worker_df["Input.label"] == 1).sum()
        n_negative_labels = (worker_df["Input.label"] == 0).sum()
        n_positive_labels_agree = ((worker_df["Input.label"] == 1)
                                   & ((worker_df["Answer.portray"] == "yes")
                                      | (worker_df["Answer.portray"] == "maybeyes"))).sum()
        n_negative_labels_agree = ((worker_df["Input.label"] == 0)
                                   & ((worker_df["Answer.portray"] == "no")
                                      | (worker_df["Answer.portray"] == "maybeno"))).sum()
        performance_on_positive_labels = (100 * n_positive_labels_agree / n_positive_labels
                                          if n_positive_labels else np.nan)
        performance_on_negative_labels = (100 * n_negative_labels_agree / n_negative_labels
                                          if n_negative_labels else np.nan)
        overall_performance = (100 * (n_positive_labels_agree + n_negative_labels_agree) /
                               (n_positive_labels + n_negative_labels))
        worktime_arr = worker_df["WorkTimeInSeconds"]/60
        avg_worktime = np.mean(worktime_arr)
        std_worktime = np.std(worktime_arr)
        median_worktime = np.median(worktime_arr)
        n_yes = (worker_df["Answer.portray"] == "yes").sum()
        n_maybeyes = (worker_df["Answer.portray"] == "maybeyes").sum()
        n_notsure = (worker_df["Answer.portray"] == "notsure").sum()
        n_maybeno = (worker_df["Answer.portray"] == "maybeno").sum()
        n_no = (worker_df["Answer.portray"] == "no").sum()
        p_yes = 100 * n_yes / n_ratings
        p_maybeyes = 100 * n_maybeyes / n_ratings
        p_notsure = 100 * n_notsure / n_ratings
        p_maybeno = 100 * n_maybeno / n_ratings
        p_no = 100 * n_no / n_ratings
        worker_rows.append([worker_id, n_ratings, n_positive_labels, n_negative_labels, performance_on_positive_labels,
                            performance_on_negative_labels, overall_performance, avg_worktime, std_worktime,
                            median_worktime, n_yes, n_maybeyes, n_notsure, n_maybeno, n_no,
                            p_yes, p_maybeyes, p_notsure, p_maybeno, p_no])

    worker_df = pd.DataFrame(worker_rows,
                             columns=["WorkerId", "nratings", "npositive", "nnegative",
                                      "perfpositive", "perfnegative", "perf",
                                      "meanworktime", "stdworktime", "medianworktime",
                                      "nyes", "nmaybeyes", "nnotsure", "nmaybeno", "nno",
                                      "percentyes", "percentmaybeyes", "percentnotsure", "percentmaybeno", "percentno"])
    worker_df.to_csv(os.path.join(FLAGS.data_dir, FLAGS.o), index=False)

if __name__ == '__main__':
    app.run(calculate_worker_performance)