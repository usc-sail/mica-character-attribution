"""Replace character names with character ids in the text"""
import data_utils

import os
import pandas as pd
import tqdm

def anonymize():
    movie_scripts_dir = os.path.join(data_utils.DATADIR, "movie-scripts")
    map_file = os.path.join(data_utils.DATADIR, "CHATTER/character-movie-map.csv")

    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    name_and_imdbid_to_characterid = {}
    for _, row in map_df.iterrows():
        name_and_imdbid_to_characterid[(row["name"], row["imdb-id"])] = row["character"]
    n = map_df["character"].str[1:].astype(int).max() + 1

    for imdbid in tqdm.tqdm(os.listdir(movie_scripts_dir), unit="movie"):
        segments_file = os.path.join(movie_scripts_dir, imdbid, "segments.csv")
        if os.path.exists(segments_file):
            mentions_file = os.path.join(movie_scripts_dir, imdbid, "mentions.csv")
            segments_df = pd.read_csv(segments_file, index_col=None)
            mentions_df = pd.read_csv(mentions_file, index_col=None)
            segments_df = segments_df.merge(mentions_df, how="left", on="segment-id")
            segments = []
            for _, segment_df in segments_df.groupby("segment-id", sort=False):
                segment_df = segment_df.sort_values("start", ascending=False)
                segment_text = segment_df["segment-text"].values[0]
                for _, row in segment_df.iterrows():
                    if pd.notna(row["imdb-character"]):
                        if (row["imdb-character"], imdbid) in name_and_imdbid_to_characterid:
                            characterid = name_and_imdbid_to_characterid[(row["imdb-character"], imdbid)]
                        else:
                            characterid = "C" + str(n).zfill(4)
                            name_and_imdbid_to_characterid[(row["imdb-character"], imdbid)] = characterid
                            n += 1
                        charactername = "CHARACTER" + characterid[1:]
                        start, end = int(row["start"]), int(row["end"])
                        segment_text = segment_text[:start] + charactername + segment_text[end:]
                segments.append(segment_text)
            anonymized_script = "\n".join(segments)
            anonymized_script_file = os.path.join(movie_scripts_dir, imdbid, "anonymized-script.txt")
            with open(anonymized_script_file, "w") as fw:
                fw.write(anonymized_script)

if __name__ == '__main__':
    anonymize()