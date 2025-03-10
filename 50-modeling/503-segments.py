"""Extract character segments from script"""
import data_utils

from absl import app
from absl import flags
import numpy as np
import os
import pandas as pd
import re
import tqdm

flags.DEFINE_bool("anonymize", default=False, help="anonymize character names")
FLAGS = flags.FLAGS

def write_statistics(arr, name, fw):
    mean = np.mean(arr)
    _max = max(arr)
    _min = min(arr)
    median = np.median(arr)
    std = np.std(arr)
    fw.write(f"{name}:\n")
    fw.write(f"mean = {mean:.2f}\n")
    fw.write(f"std = {std:.2f}\n")
    fw.write(f"min = {_min}\n")
    fw.write(f"max = {_max}\n")
    fw.write(f"median = {median:.2f}\n")
    for f in [0.8, 0.9, 0.95, 0.98, 0.99]:
        fw.write(f"{f} %tile = {np.quantile(arr, f):.2f}\n")
    fw.write("\n")

def extract_segments(_):
    """Extract segments from movie script where character is mentioned"""
    # get file paths
    movie_scripts_dir = os.path.join(data_utils.DATADIR, "movie-scripts")
    map_file = os.path.join(data_utils.DATADIR, "CHATTER/character-movie-map.csv")
    output_dir = os.path.join(data_utils.DATADIR,
                              "50-modeling",
                              "anonymized-segments" if FLAGS.anonymize else "segments")
    os.makedirs(output_dir, exist_ok=True)

    # read data
    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    imdbid_to_segments_df = {}
    imdbid_to_mentions_df = {}
    for imdbid in tqdm.tqdm(map_df["imdb-id"].unique(), desc="read movie scripts"):
        segments_file = os.path.join(movie_scripts_dir, imdbid, "segments.csv")
        mentions_file = os.path.join(movie_scripts_dir, imdbid, "mentions.csv")
        segments_df = pd.read_csv(segments_file, index_col=None)
        mentions_df = pd.read_csv(mentions_file, index_col=None)
        imdbid_to_segments_df[imdbid] = segments_df
        imdbid_to_mentions_df[imdbid] = mentions_df

    # anonymize segments
    if FLAGS.anonymize:
        name_and_imdbid_to_characterid = {}
        for _, row in map_df.iterrows():
            name_and_imdbid_to_characterid[(row["name"], row["imdb-id"])] = row["character"]
        n = map_df["character"].str[1:].astype(int).max() + 1
        for imdbid in tqdm.tqdm(imdbid_to_segments_df.keys(), desc="anonymizing"):
            segments_df = imdbid_to_segments_df[imdbid]
            mentions_df = imdbid_to_mentions_df[imdbid]
            segments_df = segments_df.merge(mentions_df, how="left", on="segment-id")
            rows = []
            for segment_id, segment_df in segments_df.groupby("segment-id", sort=False):
                if pd.notna(segment_df["imdb-character"].values[0]):
                    segment_df = segment_df.sort_values("start", ascending=False)
                    segment_text = segment_df["segment-text"].values[0]
                    for _, row in segment_df.iterrows():
                        name, start, end = row["imdb-character"], int(row["start"]), int(row["end"])
                        if (name, imdbid) in name_and_imdbid_to_characterid:
                            characterid = name_and_imdbid_to_characterid[(name, imdbid)]
                        else:
                            characterid = "C" + str(n).zfill(4)
                            name_and_imdbid_to_characterid[(row["imdb-character"], imdbid)] = characterid
                            n += 1
                        charactername = "CHARACTER" + characterid[1:]
                        segment_text = segment_text[:start] + charactername + segment_text[end:]
                    rows.append([segment_id, segment_text])
                else:
                    rows.append([segment_id, segment_df["segment-text"].values[0]])
            segments_df = pd.DataFrame(rows, columns=["segment-id", "segment-text"])
            imdbid_to_segments_df[imdbid] = segments_df

    # process data
    paragraph_sizes = []
    segment_sizes = []
    character_sizes = []
    n_paragraphs_per_segment = []
    n_segments_per_character = []
    n_paragraphs_per_character = []
    for characterid, character_df in tqdm.tqdm(map_df.groupby("character"), desc="find character segments", 
                                               total=len(map_df["character"].unique()), unit="character"):
        n_character_segments = 0
        n_character_paragraphs = 0
        character_size = 0
        for imdbid, name in character_df[["imdb-id", "name"]].itertuples(index=False, name=None):
            segments_df, mentions_df = imdbid_to_segments_df[imdbid], imdbid_to_mentions_df[imdbid]
            character_segmentids = set(mentions_df.loc[mentions_df["imdb-character"] == name, "segment-id"])
            if character_segmentids:
                n_character_segments += 1
                character_segment_paragraphs = []
                segmentids = segments_df["segment-id"].tolist()
                segment_texts = segments_df["segment-text"].tolist()
                i = 0
                while i < len(segmentids):
                    if segmentids[i] in character_segmentids:
                        j = i + 1
                        while j < len(segmentids) and segmentids[j] in character_segmentids:
                            j = j + 1
                        paragraph = " ".join(segment_texts[i: j])
                        paragraph = re.sub("\s+", " ", paragraph).strip()
                        paragraph_sizes.append(len(paragraph.split()))
                        n_character_paragraphs += 1
                        character_segment_paragraphs.append(paragraph)
                        i = j
                    else:
                        i = i + 1
                n_paragraphs_per_segment.append(len(segment_texts))
                character_segments_file = os.path.join(output_dir, f"{characterid}-{imdbid}.txt")
                character_segments_text = "\n".join(character_segment_paragraphs)
                segment_size = len(character_segments_text.split())
                segment_sizes.append(segment_size)
                character_size += segment_size
                with open(character_segments_file, "w") as fw:
                    fw.write(character_segments_text)
        character_sizes.append(character_size)
        n_segments_per_character.append(n_character_segments)
        n_paragraphs_per_character.append(n_character_paragraphs)

    # find statistics
    stats_file = os.path.join(output_dir, "stats.txt")
    with open(stats_file, "w") as fw:
        write_statistics(paragraph_sizes, "words per paragraph", fw)
        write_statistics(segment_sizes, "words per segment", fw)
        write_statistics(character_sizes, "words per character", fw)
        write_statistics(n_paragraphs_per_segment, "paragraphs per segment", fw)
        write_statistics(n_segments_per_character, "segments per character", fw)
        write_statistics(n_paragraphs_per_character, "paragraphs per character", fw)

if __name__ == '__main__':
    app.run(extract_segments)