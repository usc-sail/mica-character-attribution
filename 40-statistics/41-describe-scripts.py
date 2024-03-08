"""Find descriptive statistics of the script dataset"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import spacy
from spacy import tokens
import tqdm
import unidecode

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, required=True, help="data directory")
flags.DEFINE_string("scripts_dir", default="movie-scripts", help="scripts directory")
flags.DEFINE_string("data_file", default="20-preprocessing/train-dev-test-splits-with-negatives.csv",
                    help="data file containing imdb ids")
flags.DEFINE_string("output_dir", default="40-statistics/script-statistics",
                    help="output directory where stats are saved")

def find_statistics_from_arr(arr, id_arr, name, output_dir):
    arr = np.array(arr)
    mean, std, median, _max, _min = np.mean(arr), np.std(arr), np.median(arr), np.max(arr), np.min(arr)
    max_id = id_arr[np.argmax(arr)]
    min_id = id_arr[np.argmin(arr)]
    q90, q95, q99 = np.quantile(arr, 0.9), np.quantile(arr, 0.95), np.quantile(arr, 0.99)
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.hist(arr, bins=50)
    plt.title(f"distribution of {name}")
    plt.subplot(1, 2, 2)
    plt.hist(arr[arr < q95], bins=50)
    plt.title(f"distribution of {name} (<95%tile)")
    histogram_file = os.path.join(output_dir, f"{name}-histogram.png")
    statistics_file = os.path.join(output_dir, f"{name}-statistics.txt")
    plt.savefig(histogram_file)
    with open(statistics_file, "w") as fw:
        fw.write(f"{name} statistics =>\n")
        fw.write(f"\tmean    = {mean:.2f}\n")
        fw.write(f"\tstd     = {std:.2f}\n")
        fw.write(f"\tmedian  = {median:.1f}\n")
        fw.write(f"\tmax     = {_max} ({max_id})\n")
        fw.write(f"\tmin     = {_min} ({min_id})\n")
        fw.write(f"\t90%tile = {q90:.1f}\n")
        fw.write(f"\t95%tile = {q95:.1f}\n")
        fw.write(f"\t99%tile = {q99:.1f}")

def find_statistics(_):
    data_dir = FLAGS.data_dir
    scripts_dir = FLAGS.scripts_dir
    data_file = os.path.join(data_dir, FLAGS.data_file)
    output_dir = os.path.join(data_dir, FLAGS.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_df = pd.read_csv(data_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
    imdb_ids = data_df["imdb-id"].unique()
    nlp = spacy.blank("en")

    segment_sizes = []
    script_sizes = []
    n_segments_per_movie = []
    segment_ids = []

    for imdb_id in tqdm.tqdm(imdb_ids):
        segments_file = os.path.join(scripts_dir, imdb_id, "segments.csv")
        spacy_segments_file = os.path.join(scripts_dir, imdb_id, "spacy-segments.bytes")
        segments_df = pd.read_csv(segments_file, index_col=None)
        with open(spacy_segments_file, "rb") as fr:
            doc_bin = tokens.DocBin().from_bytes(fr.read())
        docs = list(doc_bin.get_docs(nlp.vocab))
        assert len(docs) == len(segments_df)

        script_size = 0
        for segment_id, doc in zip(segments_df["id"], docs):
            segment_size = len(doc)
            segment_ids.append(segment_id)
            segment_sizes.append(segment_size)
            script_size += segment_size
        script_sizes.append(script_size)
        n_segments_per_movie.append(len(docs))

    find_statistics_from_arr(segment_sizes, segment_ids, "segment-size", output_dir)
    find_statistics_from_arr(script_sizes, imdb_ids, "script-size", output_dir)
    find_statistics_from_arr(n_segments_per_movie, imdb_ids, "n-segments-per-movie", output_dir)

if __name__ == '__main__':
    app.run(find_statistics)