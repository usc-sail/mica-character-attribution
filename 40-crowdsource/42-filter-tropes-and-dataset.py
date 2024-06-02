"""Parse the response of prompting GPT for trope summary. Remove tropes that cannot be portrayed by a character.
Also remove dataset rows containing such tropes."""
import os
import re
import spacy
import pandas as pd
from spacy.tokenizer import Tokenizer

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory")
flags.DEFINE_string("dataset_file", default="60-modeling/dataset.csv", help="old dataset csv file")
flags.DEFINE_string("new_dataset_file", default="60-modeling/dataset-with-only-character-tropes.csv",
                    help="new dataset csv file only containing rows for tropes that can be portrayed by a character")
flags.DEFINE_string("new_tropes_file", default="60-modeling/character-tropes.csv",
                    help="new tropes csv file only containing tropes that can be portrayed by a character")
flags.DEFINE_string("prompt_response_file", default="60-modeling/summary-response.csv",
                    help="csv file containing the output response of prompting GPT to summarize trope definition")

def filter_tropes_and_dataset(_):
    data_dir = FLAGS.data_dir
    dataset_file = os.path.join(data_dir, FLAGS.dataset_file)
    new_dataset_file = os.path.join(data_dir, FLAGS.new_dataset_file)
    new_tropes_file = os.path.join(data_dir, FLAGS.new_tropes_file)
    response_file = os.path.join(data_dir, FLAGS.prompt_response_file)

    nlp = spacy.blank("en")
    tokenizer = Tokenizer(nlp.vocab)

    dataset_df = pd.read_csv(dataset_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
    responses_df = pd.read_csv(response_file, index_col=None)
    is_single_trope_arr = []
    can_portray_trope_arr = []
    summary_arr = []

    for response in responses_df["response"]:
        response_text = response.strip()
        lines = response_text.split("\n")
        first_line = lines[0]
        second_line = lines[1]
        summary = " ".join(lines[2:])
        is_single_trope = re.sub(r"[^a-zA-Z]", "", first_line).lower() == "yes"
        can_portray_trope = re.sub(r"[^a-zA-Z]", "", second_line).lower() == "yes"
        summary = re.sub(r"^\s*3[^a-zA-Z]*", "", summary)
        summary = re.sub(r"\s+", " ", summary)
        is_single_trope_arr.append(is_single_trope)
        can_portray_trope_arr.append(can_portray_trope)
        summary_arr.append(summary)

    responses_df["is-single-trope"] = is_single_trope_arr
    responses_df["can-portray"] = can_portray_trope_arr
    responses_df["summary"] = summary_arr
    responses_df["definition-ntokens"] = responses_df["definition"].apply(lambda text: len(tokenizer(text)))
    responses_df["summary-ntokens"] = responses_df["summary"].apply(lambda text: len(tokenizer(text)))
    n_character_tropes = responses_df["can-portray"].sum()
    character_tropes = set(responses_df.loc[responses_df["can-portray"], "trope"].tolist())
    print(f"{n_character_tropes}/{len(responses_df)} are character tropes")

    n_old = len(dataset_df)
    n_pos_old = (dataset_df["label"] == 1).sum()
    n_neg_old = (dataset_df["label"] == 0).sum()
    dataset_df = dataset_df[dataset_df["trope"].isin(character_tropes)]
    n_new = len(dataset_df)
    n_pos_new = (dataset_df["label"] == 1).sum()
    n_neg_new = (dataset_df["label"] == 0).sum()
    print(f"{n_old} rows in old dataset ({n_pos_old} +ve + {n_neg_old} -ve)")
    print(f"{n_new} rows in new dataset ({n_pos_new} +ve + {n_neg_new} -ve)")
    n_characters = dataset_df["character"].unique().size
    n_movies = dataset_df["imdb-id"].unique().size
    n_tropes = dataset_df["trope"].unique().size
    n_components = dataset_df["component"].unique().size
    print(f"{n_characters} characters, {n_movies} movies, {n_tropes} tropes, and {n_components} components in new"
          " dataset")
    responses_df.to_csv(new_tropes_file, index=False)
    dataset_df.to_csv(new_dataset_file, index=False)

if __name__ == '__main__':
    app.run(filter_tropes_and_dataset)