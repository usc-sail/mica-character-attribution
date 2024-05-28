"""Create and save character tensors"""
from dataloaders import tensorize

import os
import pandas as pd
import torch
import tqdm
from transformers import AutoTokenizer

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("tokenizer", default=None, help="huggingface tokenizer name", required=True)
flags.DEFINE_string("data_file", default="20-preprocessing/train-dev-test-splits-with-negatives.csv",
                    help="data file")
flags.DEFINE_string("scripts_dir", default="movie-scripts", help="movie scripts directory")
flags.DEFINE_string("tensors_dir", default="60-modeling/tensors/character", 
                    help="directory to save character tensors directory")

def save_character_tensors(_):
    data_dir = FLAGS.data_dir
    tokenizer_model = FLAGS.tokenizer
    data_file = os.path.join(data_dir, FLAGS.data_file)
    scripts_dir = os.path.join(data_dir, FLAGS.scripts_dir)
    tensors_dir = os.path.join(data_dir, FLAGS.tensors_dir, tokenizer_model)
    data_df = pd.read_csv(data_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True, add_prefix_space=True)
    log_dir = os.path.join(tensors_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(program_name="save-character-tensors", log_dir=log_dir)
    logging.get_absl_handler().setFormatter(None)

    imdb_ids = []
    imdb_characters_arr = []
    segments_dfs = []
    mentions_dfs = []
    utterances_dfs = []

    for imdb_id, imdb_df in tqdm.tqdm(data_df.groupby("imdb-id"), total=data_df["imdb-id"].unique().size,
                                      desc="reading segments and spans"):
        imdb_characters = imdb_df["imdb-character"].unique()
        segments_file = os.path.join(scripts_dir, imdb_id, "segments.csv")
        mentions_file = os.path.join(scripts_dir, imdb_id, "mentions.csv")
        utterances_file = os.path.join(scripts_dir, imdb_id, "utterances.csv")
        segments_df = pd.read_csv(segments_file, index_col=None)
        mentions_df = pd.read_csv(mentions_file, index_col=None)
        utterances_df = pd.read_csv(utterances_file, index_col=None)
        imdb_ids.append(imdb_id)
        imdb_characters_arr.append(imdb_characters)
        segments_dfs.append(segments_df)
        mentions_dfs.append(mentions_df)
        utterances_dfs.append(utterances_df)

    for i, (imdb_id, imdb_characters, segments_df, mentions_df, utterances_df) in enumerate(zip(
                imdb_ids, imdb_characters_arr, segments_dfs, mentions_dfs, utterances_dfs)):
        imdb_characters = list(set(imdb_characters).intersection(
            mentions_df["imdb-character"].unique().tolist() + utterances_df["imdb-character"].unique().tolist()))
        for imdb_character in imdb_characters:
            logging.info(f"{i + 1:3d}/{len(imdb_ids)}. imdb-id = {imdb_id}, character = '{imdb_character}'")
            try:
                token_ids, mentions_mask, utterances_mask, names_mask, _ = tensorize.create_tensors(
                    tokenizer, [imdb_character], segments_df, mentions_df, utterances_df,
                    tensorize_character_segments_only=True)
                imdb_tensors_dir = os.path.join(tensors_dir, imdb_id, imdb_character)
                os.makedirs(imdb_tensors_dir, exist_ok=True)
                torch.save(token_ids, os.path.join(imdb_tensors_dir, "token-ids.pt"))
                torch.save(mentions_mask, os.path.join(imdb_tensors_dir, "mentions-mask.pt"))
                torch.save(utterances_mask.squeeze(), os.path.join(imdb_tensors_dir, "utterances-mask.pt"))
                torch.save(names_mask.squeeze(), os.path.join(imdb_tensors_dir, "names-mask.pt"))
                logging.info("done")
            except Exception as e:
                logging.warning(str(e))
            logging.info("")

if __name__ == '__main__':
    app.run(save_character_tensors)