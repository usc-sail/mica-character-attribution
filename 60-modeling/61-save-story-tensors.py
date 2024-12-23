"""Create and save story tensors"""
from dataloaders import tensorize
from models import pretrained
import datadirs

import os
import pandas as pd
import torch
import tqdm

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_enum("model", default=None, enum_values=["roberta", "longformer"], help="model")

def save_story_tensors(_):
    model = FLAGS.model
    datadir = datadirs.datadir
    mapfile = os.path.join(datadir, "CHATTER/character-movie-map.csv")
    scriptsdir = os.path.join(datadir, "movie-scripts")
    tensorsdir = os.path.join(datadir, "60-modeling/tensors/story", model)
    map_df = pd.read_csv(mapfile, index_col=None, dtype=str)
    if model == "roberta":
        tokenizer = pretrained.Tokenizer.roberta()
    elif model == "longformer":
        tokenizer = pretrained.Tokenizer.longformer()
    log_dir = os.path.join(tensorsdir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(program_name="save-story-tensors", log_dir=log_dir)
    logging.get_absl_handler().setFormatter(None)

    imdb_ids = []
    imdb_characters_arr = []
    segments_dfs = []
    mentions_dfs = []
    utterances_dfs = []

    for imdb_id, imdb_df in tqdm.tqdm(map_df.groupby("imdb-id"), total=map_df["imdb-id"].unique().size,
                                      desc="reading movie segments and spans"):
        imdb_characters = imdb_df["name"].unique()
        segments_file = os.path.join(scriptsdir, imdb_id, "segments.csv")
        mentions_file = os.path.join(scriptsdir, imdb_id, "mentions.csv")
        utterances_file = os.path.join(scriptsdir, imdb_id, "utterances.csv")
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
        logging.info(f"{i + 1:3d}/{len(imdb_ids)}\nimdb-id = {imdb_id}")
        try:
            token_ids, attention_mask, mentions_idx, utterances_idx, names_idx = tensorize.create_tensors(
                tokenizer, imdb_characters, segments_df, mentions_df, utterances_df)
            imdb_tensors_dir = os.path.join(tensorsdir, imdb_id)
            os.makedirs(imdb_tensors_dir, exist_ok=True)
            torch.save(token_ids, os.path.join(imdb_tensors_dir, "token-ids.pt"))
            torch.save(attention_mask, os.path.join(imdb_tensors_dir, "mask.pt"))
            torch.save(mentions_idx, os.path.join(imdb_tensors_dir, "mentions-idx.pt"))
            torch.save(utterances_idx, os.path.join(imdb_tensors_dir, "utterances-idx.pt"))
            torch.save(names_idx, os.path.join(imdb_tensors_dir, "names-idx.pt"))
            with open(os.path.join(imdb_tensors_dir, "characters.txt"), "w") as fw:
                fw.write("\n".join(imdb_characters))
            logging.info(f"\ttoken-ids      = {tuple(token_ids.shape)}")
            logging.info(f"\tattention-mask = {tuple(attention_mask.shape)}")
            logging.info(f"\tnames-idx      = {tuple(names_idx.shape)}")
            logging.info(f"\tmentions-idx   = {tuple(mentions_idx.shape)}")
            logging.info(f"\tutterances-idx = {tuple(utterances_idx.shape)}")
        except Exception as e:
            logging.warning(str(e))
        logging.info("")

if __name__ == '__main__':
    app.run(save_story_tensors)