"""Find interrater agreement and rater-dataset agreement"""
import os
import numpy as np
import pandas as pd

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("crowdsource_response_file", default=None, help="full human judgements response csv file",
                    required=True)

def analyze_interrater_and_rater_dataset_agreement(_):
    data_dir = FLAGS.data_dir
    annotations_file = os.path.join(data_dir, FLAGS.crowdsource_response_file)
    annotations_df = pd.read_csv(annotations_file, index_col=None)
    n = annotations_df.loc[~annotations_df["_golden"], "_unit_id"].unique().size
    m_arr = annotations_df[~annotations_df["_golden"]].groupby("_unit_id").agg(len)["label"].unique()
    assert len(m_arr) == 1, "number of judgements per sample is uneven across samples"
    m = m_arr[0]
    ann_mat = np.zeros((n, m), dtype=int)
    label_arr = np.full(n, fill_value=False)
    content_arr = np.full(n, fill_value=False)
    ann_text_to_value = {"yes": 2, "maybe_yes": 1, "unsure": 0, "maybe_no": -1, "no": -2}
    annotations_df["ann"] = annotations_df["question_portray"].apply(lambda x: ann_text_to_value[x])

    for i, (_, sample_df) in enumerate(annotations_df[~annotations_df["_golden"]].groupby("_unit_id")):
        ann_mat[i] = sample_df["ann"].values
        label_arr[i] = sample_df["label"].values[0] == 1
        content_arr[i] = pd.notna(sample_df["content-text"].values[0])

    n_pos = label_arr == 1
    n_neg = label_arr == 0
    n_raters_strong_agree = (ann_mat == 2).all(axis=1) | (ann_mat == -2).all(axis=1)
    n_raters_agree = (ann_mat > 0).all(axis=1) | (ann_mat < 0).all(axis=1)
    n_raters_dataset_strong_agree = (((ann_mat == 2).all(axis=1) & label_arr)
                                     | ((ann_mat == -2).all(axis=1) & ~label_arr))
    n_raters_dataset_agree = (((ann_mat > 0).all(axis=1) & label_arr) | ((ann_mat < 0).all(axis=1) & ~label_arr))
    print(f"number of samples = {n} ({n_pos.sum()} +ve + {n_neg.sum()} -ve)")
    print(f"number of samples on which raters strongly agree = {n_raters_strong_agree.sum()} "
          f"({(n_raters_strong_agree & n_pos).sum()} +ve + {(n_raters_strong_agree & n_neg).sum()} -ve)")
    print(f"number of samples on which raters agree = {n_raters_agree.sum()} "
          f"({(n_raters_agree & n_pos).sum()} +ve + {(n_raters_agree & n_neg).sum()} -ve)")
    print(f"number of samples on which raters strongly agree with dataset label = {n_raters_dataset_strong_agree.sum()}"
          f" ({(n_raters_dataset_strong_agree & n_pos).sum()} +ve + {(n_raters_dataset_strong_agree & n_neg).sum()}"
          " -ve)")
    print(f"number of samples on which raters agree with dataset label = {n_raters_dataset_agree.sum()} "
          f"({(n_raters_dataset_agree & n_pos).sum()} +ve + {(n_raters_dataset_agree & n_neg).sum()} -ve)")

if __name__ == '__main__':
    app.run(analyze_interrater_and_rater_dataset_agreement)