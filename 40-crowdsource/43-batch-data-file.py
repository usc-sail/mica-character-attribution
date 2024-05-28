"""Batch crowdsource input"""
import os
import pandas as pd

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("input_file", default="40-crowdsource/input-with-pics.csv", help="input csv file with picture urls")
flags.DEFINE_string("dataset_file", default="60-modeling/dataset.csv", help="dataset csv file containing year field")
flags.DEFINE_string("batches_dir", default="40-crowdsource/batches", help="directory where the batches are saved")

def batch_data_file(_):
    data_dir = FLAGS.data_dir
    dataset_file = os.path.join(data_dir, FLAGS.dataset_file)
    input_file = os.path.join(data_dir, FLAGS.input_file)
    batches_dir = os.path.join(data_dir, FLAGS.batches_dir)

    input_df = pd.read_csv(input_file, index_col=None)
    input_df.rename(columns={"trope": "trope-url"}, inplace=True)
    input_df["trope"] = input_df["trope-url"].str.extract(r">(\w+)<")
    character_df = pd.read_csv(dataset_file, index_col=None, usecols=["character", "year", "trope", "content-text"],
                               dtype={"content-text": str})
    character_df.rename(columns={"content-text": "actual-content-text"}, inplace=True)
    character_year_df = character_df.groupby("character").agg({"year": "min"}).reset_index()
    character_trope_df = character_df[["character", "trope", "actual-content-text"]].drop_duplicates()
    input_df = input_df.merge(character_year_df, how="left", on="character")
    input_df = input_df.merge(character_trope_df, how="left", on=["character", "trope"])
    input_df.sort_values(by=["character", "trope"], inplace=True)
    input_df = input_df.sample(frac=1, random_state=0)
    input1_df = input_df[input_df["year"] >= 2000]
    input2_df = input_df[input_df["year"] < 2000]
    input_df = pd.concat([input1_df, input2_df])
    input_df["batch"] = 10
    input_df.index = pd.RangeIndex(len(input_df))
    input_df.loc[:300, "batch"] = 0
    for i in range(10):
        input_df.loc[300 + i * 10000: 300 + (i + 1) * 10000, "batch"] = i + 1
    input_df = input_df[["character", "imdb-character-name", "character-introduction", "trope-url", "trope",
                         "content-text", "actual-content-text", "picture-1", "picture-2", "picture-3", "year", "batch",
                         "label"]]
    print("n = number of samples")
    print("p1 = percentage of samples with evidence and label=1")
    print("p2 = percentage of samples with no evidence and label=1")
    print("p3 = percentage of samples with label=0\n")
    for i in range(11):
        batch_df = input_df[input_df["batch"] == i]
        n1 = (batch_df["content-text"].notna() & (batch_df["label"] == 1)).sum()
        n2 = (batch_df["content-text"].isna() & (batch_df["label"] == 1)).sum()
        n3 = (batch_df["label"] == 0).sum()
        n = n1 + n2 + n3
        p1 = 100 * n1 / n
        p2 = 100 * n2 / n
        p3 = 100 * n3 / n
        batch_file = os.path.join(batches_dir, f"batch-{i}.csv")
        batch_df.to_csv(batch_file, index=False)
        print(f"n={n} p1={p1:3.0f}% p2={p2:3.0f}% p3={p3:3.0f}%")

if __name__ == '__main__':
    app.run(batch_data_file)