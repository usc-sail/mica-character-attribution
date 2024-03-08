"""Inject negative (character, trope) samples to complete the dataset"""
import os
import tqdm
import random
import pandas as pd

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("splits", default="20-preprocessing/train-dev-test-splits.csv",
                    help=("csv file of imdb-id, title, year, character, imdb-character, rank, tvtrope-character-url, "
                          "tvtrope-character-xpath, tvtrope-character-name, sequence-score, token-score, trope, "
                          "content-text"))
flags.DEFINE_string("output", default="20-preprocessing/train-dev-test-splits-with-negatives.csv",
                    help="csv file with same fields as splits files + a field with 0 or 1")
flags.DEFINE_string("tropes", default="20-preprocessing/tropes-with-antonyms.csv",
                    help="csv file of trope, definition, linked-tropes, antonym-tropes")

def inject_negatives(_):
    random.seed(2024)
    data_dir = FLAGS.data_dir
    splits_file = os.path.join(data_dir, FLAGS.splits)
    output_file = os.path.join(data_dir, FLAGS.output)
    tropes_file = os.path.join(data_dir, FLAGS.tropes)

    splits_df = pd.read_csv(splits_file, index_col=None, dtype={"imdb-id": str})
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    tropes = tropes_df["trope"].tolist()

    trope_to_antonyms = {}
    for _, row in tropes_df.iterrows():
        if pd.notna(row["antonym-tropes"]):
            trope_to_antonyms[row["trope"]] = set(row["antonym-tropes"].split(";"))

    character_to_tropes = {}
    character_to_negative_tropes = {}
    groups = splits_df.groupby("character")
    for character, df in tqdm.tqdm(groups, total=groups.ngroups, desc="finding character negative tropes"):
        character_tropes = set(df["trope"].tolist())
        character_to_tropes[character] = character_tropes
        character_to_negative_tropes[character] = list(filter(lambda trope: trope not in character_tropes, tropes))

    negative_df = splits_df.copy()
    negative_trope_col = []
    for _, row in tqdm.tqdm(splits_df.iterrows(), total=len(splits_df), desc="creating negatives"):
        character, trope = row["character"], row["trope"]
        antonym_tropes = set()
        if trope in trope_to_antonyms:
            antonym_tropes = trope_to_antonyms[trope]
        antonym_tropes.difference_update(character_to_tropes[character])
        antonym_tropes = sorted(antonym_tropes)
        if antonym_tropes:
            if random.random() < 0.5:
                negative_trope = random.choice(antonym_tropes)
            else:
                negative_trope = random.choice(character_to_negative_tropes[character])
        else:
            negative_trope = random.choice(character_to_negative_tropes[character])
        negative_trope_col.append(negative_trope)
    negative_df["trope"] = negative_trope_col
    negative_df.drop_duplicates(subset=["imdb-id", "character", "trope"], inplace=True)

    splits_df["label"] = 1
    negative_df["label"] = 0
    negative_df["content-text"] = ""
    splits_with_negative_df = pd.concat([splits_df, negative_df])

    total_n_movies = splits_with_negative_df["imdb-id"].unique().size
    total_n_characters = splits_with_negative_df["character"].unique().size
    total_n_tropes = splits_with_negative_df["trope"].unique().size
    total_n_samples = len(splits_with_negative_df)
    total_n_pos_samples = splits_with_negative_df["label"].sum()
    total_n_neg_samples = total_n_samples - total_n_pos_samples
    print(f"full : {total_n_movies:4d} movies, {total_n_characters:4d} characters, {total_n_tropes:4d} tropes, "
          f"{total_n_samples:6d} samples, {total_n_pos_samples} +ve + {total_n_neg_samples} -ve samples")
    for partition in ["train", "dev", "test"]:
        mask = splits_with_negative_df["partition"] == partition
        n_movies = splits_with_negative_df.loc[mask, "imdb-id"].unique().size
        n_characters = splits_with_negative_df.loc[mask, "character"].unique().size
        n_tropes = splits_with_negative_df.loc[mask, "trope"].unique().size
        n_samples = (splits_with_negative_df["partition"] == partition).sum()
        n_pos_samples = splits_with_negative_df.loc[mask, "label"].sum()
        n_neg_samples = n_samples - n_pos_samples
        percentage_movies = 100 * n_movies / total_n_movies
        percentage_characters = 100 * n_characters / total_n_characters
        percentage_tropes = 100 * n_tropes / total_n_tropes
        percentage_samples = 100 * n_samples / total_n_samples
        print(f"{partition:5s}: "
              f"{n_movies:4d} movies ({percentage_movies:.1f}%), "
              f"{n_characters:4d} characters ({percentage_characters:.1f}%), "
              f"{n_tropes:4d} tropes ({percentage_tropes:.1f}%), "
              f"{n_samples:6d} samples ({percentage_samples:.1f}%), "
              f"{n_pos_samples} +ve + {n_neg_samples} -ve samples")

    splits_with_negative_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    app.run(inject_negatives)