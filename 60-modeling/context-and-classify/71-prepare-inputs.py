"""Prepare the text input for the classification tasks"""
import datadirs

from absl import app
from absl import flags
import collections
import json
import numpy as np
import os
import pandas as pd
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_multi_integer("context", default=[0, 1, 5, 10, 20], help="number of neighboring segments")

def find_context_ix(ix, full_ix, is_slugline, context_size):
    n = len(full_ix)
    mask = np.zeros((n,), dtype=int)
    for i in ix:
        start = i
        if not is_slugline[i]:
            j = 1
            while j <= context_size and i - j >= 0 and not is_slugline[i - j] and not mask[i - j]:
                j += 1
            start = i - j + 1
        j = 1
        while j <= context_size and i + j < n and not is_slugline[i + j] and not mask[i + j]:
            j += 1
        end = i + j
        mask[start: end] = 1
    return mask.nonzero()[0]

def concatenate_adjoining_texts(texts, ix):
    adjoined_texts = []
    i = 0
    while i < len(ix):
        j = i + 1
        while j < len(ix) and ix[j] == ix[j - 1] + 1:
            j += 1
        adjoined_texts.append(" ".join(texts[i: j]))
        i = j
    return adjoined_texts

def prepare_input(_):
    map_file = os.path.join(datadirs.datadir, "CHATTER/character-movie-map.csv")
    movie_scripts_dir = os.path.join(datadirs.datadir, "movie-scripts")
    context_sizes = FLAGS.context

    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    data = {}
    for imdbid in tqdm.tqdm(map_df["imdb-id"].unique(), unit="movie", desc="read movie data"):
        segments_file = os.path.join(movie_scripts_dir, imdbid, "segments.csv")
        mentions_file = os.path.join(movie_scripts_dir, imdbid, "mentions.csv")
        segments_df = pd.read_csv(segments_file, index_col=None)
        mentions_df = pd.read_csv(mentions_file, index_col=None)
        data[imdbid] = {"segments": segments_df, "mentions": mentions_df}

    desc_contexts = {}
    utter_contexts = {}
    contexts = {}
    for context_size in context_sizes:
        desc_contexts[context_size] = collections.defaultdict(list)
        utter_contexts[context_size] = collections.defaultdict(list)
        contexts[context_size] = collections.defaultdict(list)

    for characterid, character_df in tqdm.tqdm(map_df.groupby("character"), unit="character", desc="prepare input"):
        for imdbid, name in character_df[["imdb-id", "name"]].itertuples(index=False, name=None):
            segments_df = data[imdbid]["segments"]
            mentions_df = data[imdbid]["mentions"]
            segmentids = set(mentions_df.loc[mentions_df["imdb-character"] == name, "segment-id"])
            if not segmentids:
                break
            ix = segments_df.index[segments_df["segment-id"].isin(segmentids)].values
            character_segments_df = segments_df.iloc[ix]
            utter_ix = character_segments_df[character_segments_df["segment-type"] == "utter"].index.values
            desc_ix = ix[~np.isin(ix, utter_ix)]
            story_ix = segments_df.index.values
            is_slugline = segments_df["segment-type"] == "slugline"
            for context_size in context_sizes:
                cx = find_context_ix(ix, story_ix, is_slugline, context_size)
                ux = find_context_ix(utter_ix, story_ix, is_slugline, context_size)
                dx = find_context_ix(desc_ix, story_ix, is_slugline, context_size)
                texts = segments_df.loc[cx, "segment-text"].tolist()
                utter_texts = segments_df.loc[ux, "segment-text"].tolist()
                desc_texts = segments_df.loc[dx, "segment-text"].tolist()
                texts = concatenate_adjoining_texts(texts, cx)
                utter_texts = concatenate_adjoining_texts(utter_texts, ux)
                desc_texts = concatenate_adjoining_texts(desc_texts, dx)
                contexts[context_size][characterid].extend(texts)
                if utter_texts:
                    utter_contexts[context_size][characterid].extend(utter_texts)
                if desc_texts:
                    desc_contexts[context_size][characterid].extend(desc_texts)

    for context_size in tqdm.tqdm(context_sizes, unit="context", desc="write input"):
        data_file = os.path.join(datadirs.datadir, f"70-classification/contexts/context-{context_size}.json")
        utter_data_file = os.path.join(datadirs.datadir,
                                       f"70-classification/contexts/utter-context-{context_size}.json")
        desc_data_file = os.path.join(datadirs.datadir,
                                      f"70-classification/contexts/desc-context-{context_size}.json")
        json.dump(contexts[context_size], open(data_file, "w"))
        json.dump(dict(utter_contexts[context_size]), open(utter_data_file, "w"))
        json.dump(dict(desc_contexts[context_size]), open(desc_data_file, "w"), indent=2)

    movie_data = {}
    for imdbid in map_df["imdb-id"].unique():
        script_file = os.path.join(datadirs.datadir, f"movie-scripts/{imdbid}/script.txt")
        movie_data[imdbid] = open(script_file).read().strip()
    movie_data_file = os.path.join(datadirs.datadir, f"70-classification/contexts/story.json")
    json.dump(movie_data, open(movie_data_file, "w"), indent=2)

if __name__ == '__main__':
    app.run(prepare_input)