"""Select dataset rows for which you want to collect annotations"""
import os
import pandas as pd

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("introductions_file", default="40-crowdsource/character-introductions.csv",
                    help="dataset csv file only containing character introductions")
flags.DEFINE_string("urls_file", default="40-crowdsource/character-picture-urls.csv",
                    help="csv file containing fields for character and picture urls")
flags.DEFINE_string("tropes_file", default="60-modeling/character-tropes.csv",
                    help="character tropes csv file containing a field on the summary of the trope's definition")
flags.DEFINE_integer("n", default=4000, help="number of rows to sample", lower_bound=0)
flags.DEFINE_string("crowdsource_file", default="40-crowdsource/dataset-to-annotate.csv",
                    help=("dataset csv file with character introduction, picture urls, and summary fields to"
                          " enqueue for annotation"))

def select_crowdsource_rows(_):
    data_dir = FLAGS.data_dir
    introductions_file = os.path.join(data_dir, FLAGS.introductions_file)
    urls_file = os.path.join(data_dir, FLAGS.urls_file)
    tropes_file = os.path.join(data_dir, FLAGS.tropes_file)
    n = FLAGS.n
    crowdsource_file = os.path.join(data_dir, FLAGS.crowdsource_file)

    introductions_df = pd.read_csv(introductions_file, index_col=None)
    urls_df = pd.read_csv(urls_file, index_col=None)
    tropes_df = pd.read_csv(tropes_file, index_col=None, usecols=["trope", "summary"])
    toannotate_df = introductions_df.merge(urls_df, how="left", on="character").merge(tropes_df, how="left", on="trope")
    toannotate_df.sort_values(by=["character", "trope"], inplace=True)
    toannotate_df = toannotate_df[toannotate_df["year"] >= 2010].sample(n=n, random_state=2024, replace=False)
    n_pos = (toannotate_df["label"] == 1).sum()
    n_neg = n - n_pos
    print(f"{n} samples, {n_pos} +ve and {n_neg} -ve a/c tvtrope label")
    toannotate_df.to_csv(crowdsource_file, index=False)

if __name__ == '__main__':
    app.run(select_crowdsource_rows)