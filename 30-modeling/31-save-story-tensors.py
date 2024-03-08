"""Create and save story tensors"""
from dataloaders import story

import os
import pandas as pd
import torch
from transformers import AutoTokenizer

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("tokenizer", default=None, help="huggingface tokenizer name", required=True)
flags.DEFINE_string("data_file", default="20-preprocessing/train-dev-test-splits-with-negatives.csv",
                    help="data file")
flags.DEFINE_string("scripts_dir", default="movie-scripts", help="movie scripts directory")
flags.DEFINE_string("tensors_dir", default="30-modeling/tensors/story", 
                    help="directory to save story tensors directory")

def save_story_tensors(_):
    data_dir = FLAGS.data_dir
    tokenizer_model = FLAGS.tokenizer
    data_file = os.path.join(data_dir, FLAGS.data_file)
    scripts_dir = os.path.join(data_dir, FLAGS.scripts_dir)
    tensors_dir = os.path.join(data_dir, FLAGS.tensors_dir, tokenizer_model)
    os.makedirs(tensors_dir, exist_ok=True)
    data_df = pd.read_csv(data_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True, add_prefix_space=True)
    n = 10

    imdb_ids = []
    imdb_characters_arr = []
    segments_dfs = []
    mentions_dfs = []
    utterances_dfs = []

    for imdb_id, imdb_df in data_df.groupby("imdb-id"):
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
        n -= 1
        if n == 0:
            break

    # story_token_ids_arr = []
    # mentions_mask_arr = []
    # utterances_mask_arr = []
    # names_mask_arr = []
    # mention_character_ids_arr = []
    # utterance_character_ids_arr = []

    for imdb_id, imdb_characters, segments_df, mentions_df, utterances_df in zip(
            imdb_ids, imdb_characters_arr, segments_dfs, mentions_dfs, utterances_dfs):
        story_token_ids, mentions_mask, utterances_mask, names_mask, mention_character_ids, utterance_character_ids = (
            story.create_tensors(tokenizer, imdb_characters, segments_df, mentions_df, utterances_df, verbose=True))
        print()
        # story_token_ids_arr.append(story_token_ids)
        # mentions_mask_arr.append(mentions_mask)
        # utterances_mask_arr.append(utterances_mask)
        # names_mask_arr.append(names_mask)
        # mention_character_ids_arr.append(mention_character_ids)
        # utterance_character_ids_arr.append(utterance_character_ids)
        imdb_tensors_dir = os.path.join(tensors_dir, imdb_id)
        os.makedirs(imdb_tensors_dir, exist_ok=True)
        torch.save(story_token_ids, os.path.join(imdb_tensors_dir, "token-ids.pt"))
        torch.save(mentions_mask, os.path.join(imdb_tensors_dir, "mentions-mask.pt"))
        torch.save(utterances_mask, os.path.join(imdb_tensors_dir, "utterances-mask.pt"))
        torch.save(names_mask, os.path.join(imdb_tensors_dir, "names-mask.pt"))
        torch.save(mention_character_ids, os.path.join(imdb_tensors_dir, "mention-character-ids.pt"))
        torch.save(utterance_character_ids, os.path.join(imdb_tensors_dir, "utterance-character-ids.pt"))
        with open(os.path.join(imdb_tensors_dir, "characters.txt"), "w") as fw:
            fw.write("\n".join(imdb_characters))

if __name__ == '__main__':
    app.run(save_story_tensors)