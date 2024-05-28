import os
import pandas as pd

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, required=True, help="data directory")
flags.DEFINE_string("data_file", default="20-preprocessing/train-dev-test-splits-with-negatives.csv",
                    help="data file containing imdb ids")
flags.DEFINE_string("output_file", default="40-statistics/dataset-statistics/component-statistics.txt",
                    help="output file where component stats will be saved")

def uniq_size(arr):
    return len(arr.unique())

def find_statistics(_):
    data_dir = FLAGS.data_dir
    data_file = os.path.join(data_dir, FLAGS.data_file)
    output_file = os.path.join(data_dir, FLAGS.output_file)

    with open(output_file, "w") as fw:
        df = pd.read_csv(data_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
        cdf = df.groupby("component").agg({"character": uniq_size, "imdb-id": uniq_size, "trope": uniq_size}).describe()
        mdf = df.groupby("imdb-id").agg({"character": uniq_size, "trope": uniq_size}).describe()
        xdf = df.groupby("character").agg({"imdb-id": uniq_size, "trope": uniq_size}).describe()
        fw.write("component-stats (rounded off) =>\n")
        fw.write(str(cdf.astype(int)))
        fw.write("\n\n")
        fw.write("movie-stats (rounded off) =>\n")
        fw.write(str(mdf.astype(int)))
        fw.write("\n\n")
        fw.write("character-stats (rounded off) =>\n")
        fw.write(str(xdf.astype(int)))

if __name__ == '__main__':
    app.run(find_statistics)