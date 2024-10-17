"""Create reliable annotations for the test set"""
import os
import collections
import pandas as pd

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("datadir", default=None, help="data directory", required=True)
flags.DEFINE_string("datasetfile", default="60-modeling/dataset-with-only-character-tropes.csv", help="dataset file")
flags.DEFINE_string("aggregatefile", default="40-crowdsource/mturk-aggregate-results/aggregate.csv",
                    help="aggregate file of full crowdsourced annotations")
flags.DEFINE_string("outputreliablefile", default="60-modeling/test-dataset.csv",
                    help="test dataset file for character attribution")
flags.DEFINE_integer("minconf", default=3, help="minimum confidence required to put samples into test set")

def create_reliable_test_set(_):
    data_dir = FLAGS.datadir
    dataset_file = os.path.join(data_dir, FLAGS.datasetfile)
    agg_file = os.path.join(data_dir, FLAGS.aggregatefile)
    test_file = os.path.join(data_dir, FLAGS.outputreliablefile)
    minconf = FLAGS.minconf

    agg_df = pd.read_csv(agg_file, index_col=None, dtype={"combination": str})
    dataset_df = pd.read_csv(dataset_file, index_col=None, dtype={"imdb-id": str, "content-text": str})

    agg_df["conf"] = 2 * agg_df["yes"] + agg_df["maybeyes"] - agg_df["maybeno"] - 2 * agg_df["no"]
    agg_df = agg_df[(agg_df["conf"].abs() >= minconf) & (agg_df["n"] >= 2)].copy()

    character2movies = collections.defaultdict(set)
    for character, movie in dataset_df[["character", "imdb-id"]].drop_duplicates().itertuples(index=False, name=None):
        character2movies[character].add(movie)
    agg_df["imdb-ids"] = agg_df["character"].apply(lambda character: ";".join(sorted(character2movies[character])))
    agg_df["n-imdb-ids"] = agg_df["imdb-ids"].apply(lambda imdbids: len(imdbids.split(";")))

    agg_df = agg_df.merge(dataset_df[["character", "trope", "label"]].drop_duplicates(), how="left",
                          on=["character", "trope"])
    agg_df.rename(columns={"label": "tvtrope-label"}, inplace=True)

    n_samples = len(agg_df)
    n_characters = agg_df["character"].unique().size
    n_tropes = agg_df["trope"].unique().size
    n_movies = len(set([imdbid for imdbids in agg_df["imdb-ids"] for imdbid in imdbids.split(";")]))
    n_prompts = agg_df["n-imdb-ids"].sum()

    print("reliable test dataset:")
    print(f"{n_samples} (character, trope) pairs, {n_characters} characters, {n_tropes} tropes, {n_movies} movies, "
          f"{n_prompts} (character, trope, movie) tuples or prompts")
    agg_df.to_csv(test_file, index=False)

if __name__ == '__main__':
    app.run(create_reliable_test_set)