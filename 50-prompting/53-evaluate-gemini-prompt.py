"""Evaluate the response of gemini model on prompting for character attributes"""
import os
import numpy as np
import pandas as pd

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, required=True, help="data directory")
flags.DEFINE_string("prompt_response_file", default=None, required=True,
                    help="csv file containing the response of prompting gemini model for character attribution")

def parse_answer(response):
    if response.lower().strip().startswith("yes"):
        return 1
    elif response.lower().strip().startswith("no"):
        return 0
    else:
        return np.nan

def evaluate_response(_):
    data_dir = FLAGS.data_dir
    response_file = os.path.join(data_dir, FLAGS.prompt_response_file)
    response_df = pd.read_csv(response_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
    response_df["prediction"] = response_df["response"].apply(parse_answer)
    n_errors = response_df["prediction"].isna().sum()
    n = response_df["prediction"].notna().sum()
    n_positive = (response_df["prediction"].notna() & (response_df["label"] == 1)).sum()
    tp = (response_df["prediction"].notna() & (response_df["label"] == 1) & (response_df["prediction"] == 1)).sum()
    fp = (response_df["prediction"].notna() & (response_df["label"] == 0) & (response_df["prediction"] == 1)).sum()
    fn = (response_df["prediction"].notna() & (response_df["label"] == 1) & (response_df["prediction"] == 0)).sum()
    error_percentage = 100 * n_errors / len(response_df)
    positive_percentage = 100 * n_positive / n
    accuracy = 100 * (n - fp - fn) / n
    precision = 100 * tp / (fp + tp)
    recall = 100 * tp / (fn + tp)
    print(f"prompt erred on {error_percentage:.1f}% samples")
    print(f"performance on the rest {100 - error_percentage:.1f}% samples "
          f"({positive_percentage:.1f}% +ve + {100 - positive_percentage:.1f}% -ve): accuracy = {accuracy:.1f}%, "
          f"precision = {precision:.1f}%, recall = {recall:.1f}%")

if __name__ == '__main__':
    app.run(evaluate_response)