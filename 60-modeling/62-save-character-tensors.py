"""Create and save character tensors"""
import datadirs
from dataloaders import tensorize
from models import pretrained

import numpy as np
import os
import pandas as pd
import torch
import tqdm
from transformers import AutoTokenizer

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_enum("model", default=None, enum_values=["roberta", "longformer"], help="model")
flags.DEFINE_integer("context", default=0, help="number of neighboring segments")

def save_character_tensors(_):
    model = FLAGS.model
    context = FLAGS.context
    datadir = datadirs.datadir
    mapfile = os.path.join(datadir, "CHATTER/character-movie-map.csv")
    metadatafile = os.path.join(datadir, "CHATTER/movie-metadata.csv")
    scriptsdir = os.path.join(datadir, "movie-scripts")
    tensorsdir = os.path.join(datadir, "60-modeling/tensors/character", model)
    map_df = pd.read_csv(mapfile, index_col=None, dtype=str)
    metadata_df = pd.read_csv(metadatafile, index_col=None, dtype={"imdb-id": str})
    map_df = map_df.merge(metadata_df, how="left", on="imdb-id")
    if model == "roberta":
        tokenizer = pretrained.Tokenizer.roberta()
    elif model == "longformer":
        tokenizer = pretrained.Tokenizer.longformer()
    log_dir = os.path.join(tensorsdir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(program_name="save-character-tensors", log_dir=log_dir)
    logging.get_absl_handler().setFormatter(None)

    imdbid_data = {}
    for imdbid in tqdm.tqdm(map_df["imdb-id"].unique(), desc="reading movie segments and spans"):
        segmentsfile = os.path.join(scriptsdir, imdbid, "segments.csv")
        mentionsfile = os.path.join(scriptsdir, imdbid, "mentions.csv")
        utterancesfile = os.path.join(scriptsdir, imdbid, "utterances.csv")
        segments_df = pd.read_csv(segmentsfile, index_col=None)
        mentions_df = pd.read_csv(mentionsfile, index_col=None)
        utterances_df = pd.read_csv(utterancesfile, index_col=None)
        imdbid_data[imdbid] = dict(segments=segments_df, mentions=mentions_df, utterances=utterances_df)

    characterids = []
    character_names = []
    characters_segments_dfs = []
    characters_mentions_dfs = []
    characters_utterances_dfs = []
    for characterid, characterdf in tqdm.tqdm(map_df.groupby("character"), total=map_df["character"].unique().size,
                                              desc="creating character segments and spans"):
        characterdf = characterdf.sort_values(by="year", ascending=True)
        names = []
        character_segments_dfs = []
        character_mentions_dfs = []
        character_utterances_dfs = []
        for imdbid, name in characterdf[["imdb-id", "name"]].itertuples(index=False, name=None):
            movie_segments_df = imdbid_data[imdbid]["segments"]
            movie_mentions_df = imdbid_data[imdbid]["mentions"]
            movie_utterances_df = imdbid_data[imdbid]["utterances"]
            character_mentions_df = movie_mentions_df[movie_mentions_df["imdb-character"] == name].copy()
            character_utterances_df = movie_utterances_df[movie_utterances_df["imdb-character"] == name].copy()
            if len(character_mentions_df) + len(character_utterances_df) > 0:
                character_mention_segmentids = character_mentions_df["segment-id"].tolist()
                character_utterance_segmentids = character_utterances_df["segment-id"].tolist()
                character_segmentids = set(character_mention_segmentids + character_utterance_segmentids)
                character_segments_index = movie_segments_df[
                    movie_segments_df["segment-id"].isin(character_segmentids)].index.tolist()
                character_segments_index_mask = np.zeros(len(movie_segments_df), dtype=int)
                character_segments_index_mask[character_segments_index] = 1
                for i in character_segments_index:
                    character_segments_index_mask[max(0, i - context):
                                                  min(i + context, len(movie_segments_df) - 1) + 1] = 1
                character_segments_index = character_segments_index_mask.nonzero()[0]
                character_segments_df = movie_segments_df.iloc[character_segments_index].copy()
                character_segments_df["segment-id"] = imdbid + "-" + character_segments_df["segment-id"]
                character_mentions_df["segment-id"] = imdbid + "-" + character_mentions_df["segment-id"]
                character_utterances_df["segment-id"] = imdbid + "-" + character_utterances_df["segment-id"]
                names.append(name)
                character_segments_dfs.append(character_segments_df)
                character_mentions_dfs.append(character_mentions_df)
                character_utterances_dfs.append(character_utterances_df)
        if len(character_segments_dfs) > 0:
            characterids.append(characterid)
            namelens = [len(name) for name in names]
            i = np.argmax(namelens).item()
            character_names.append(names[i])
            character_segments_df = pd.concat(character_segments_dfs, axis=0)
            character_mentions_df = pd.concat(character_mentions_dfs, axis=0)
            character_utterances_df = pd.concat(character_utterances_dfs, axis=0)
            character_mentions_df["imdb-character"] = names[i]
            character_utterances_df["imdb-character"] = names[i]
            characters_segments_dfs.append(character_segments_df)
            characters_mentions_dfs.append(character_mentions_df)
            characters_utterances_dfs.append(character_utterances_df)

    logging.info(f"\n{len(characterids)}/{map_df["character"].unique().size} characters have some mention or "
                 "utterance\n")

    for i, (characterid, character_name, character_segments_df, character_mentions_df, character_utterances_df) in (
            enumerate(zip(characterids, character_names, characters_segments_dfs, characters_mentions_dfs, 
                          characters_utterances_dfs))):
        logging.info(f"{i + 1:4d}/{len(characterids)} character-id = {characterid}")
        try:
            token_ids, mentions_idx, utterances_idx, names_idx, _, _ = tensorize.create_tensors(
                tokenizer, [character_name], character_segments_df, character_mentions_df, character_utterances_df)
            character_tensors_dir = os.path.join(tensorsdir, characterid)
            os.makedirs(character_tensors_dir, exist_ok=True)
            torch.save(token_ids, os.path.join(character_tensors_dir, "token-ids.pt"))
            torch.save(mentions_idx, os.path.join(character_tensors_dir, "mentions-idx.pt"))
            torch.save(utterances_idx, os.path.join(character_tensors_dir, "utterances-idx.pt"))
            torch.save(names_idx, os.path.join(character_tensors_dir, "names-idx.pt"))
            logging.info(f"token-ids = {tuple(token_ids.shape)}")
            logging.info(f"names-idx = {tuple(names_idx.shape)}")
            logging.info(f"mentions-idx = {tuple(mentions_idx.shape)}")
            logging.info(f"utterances-idx = {tuple(utterances_idx.shape)}")
        except Exception as e:
            logging.warning(str(e))
        logging.info("")

if __name__ == '__main__':
    app.run(save_character_tensors)