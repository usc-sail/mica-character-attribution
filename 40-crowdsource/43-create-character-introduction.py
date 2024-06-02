"""Create csv file that contains the introduction of the character to be shown to workers in the annotation task"""
import os
import json
import tqdm
import random
import numpy as np
import pandas as pd

from absl import flags
from absl import app

random.seed(2024)
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("dataset_file", default="60-modeling/dataset-with-only-character-tropes.csv",
                    help="dataset csv file")
flags.DEFINE_string("movie_scripts_dir", default="movie-scripts", help="movie scripts dir")
flags.DEFINE_string("crowdsource_file", default="40-crowdsource/character-introductions.csv",
                    help="csv file containing character introductions")

def create_character_introduction(imdb_character_name, imdb_titles, imdb_urls, imdb_character_urls):
    intro = f"<strong>{imdb_character_name}</strong> is a character that appears in the movie"
    values = []
    for imdb_title, imdb_url, imdb_character_url in zip(imdb_titles, imdb_urls, imdb_character_urls):
        values.append(f"<a href=\"{imdb_url}\" target=\"_blank\"><i>{imdb_title}</i></a>"
                       + f" (<a href=\"{imdb_character_url}\" target=\"_blank\">Character Page</a>)"
                       if imdb_character_url else "")
    if len(values) == 1:
        intro += f" {values[0]}."
    else:
        intro += "s: " + ", ".join(values[:-1]) + f" and {values[-1]}."
    return intro

def create_data_file(_):
    data_dir = FLAGS.data_dir
    dataset_file = os.path.join(data_dir, FLAGS.dataset_file)
    movie_scripts_dir = os.path.join(data_dir, FLAGS.movie_scripts_dir)
    crowdsource_file = os.path.join(data_dir, FLAGS.crowdsource_file)
    dataset_df = pd.read_csv(dataset_file, index_col=None, dtype={"imdb-id": str, "content-text": str})

    imdb_id_to_movie_data = {}
    imdb_ids = dataset_df["imdb-id"].unique()
    for imdb_id in tqdm.tqdm(imdb_ids, desc="reading imdb data"):
        imdb_file = os.path.join(movie_scripts_dir, imdb_id, "imdb.json")
        with open(imdb_file) as fr:
            movie_data = json.load(fr)
        imdb_id_to_movie_data[imdb_id] = movie_data

    rows = []
    header = ["character", "name", "character-urls", "character-introduction", "trope", "trope-url", "content-text",
              "year", "label"]
    for cid, character_df in tqdm.tqdm(dataset_df.groupby("character", sort=True), unit="character",
                                                total=dataset_df["character"].unique().size,
                                                desc="creating character data"):
        imdb_character_names = character_df["imdb-character"].unique()
        imdb_ids = character_df["imdb-id"].unique()
        imdb_titles = [imdb_id_to_movie_data[imdb_id]["title"] for imdb_id in imdb_ids]
        imdb_years = [imdb_id_to_movie_data[imdb_id]["year"] for imdb_id in imdb_ids]
        imdb_id_urls = [f"https://www.imdb.com/title/tt{imdb_id}/" for imdb_id in imdb_ids]
        imdb_character_urls = []
        for imdb_id in imdb_ids:
            for character in imdb_id_to_movie_data[imdb_id]["cast"]:
                if character.get("character") in imdb_character_names and character.get("personID") is not None:
                    character_id = character.get("personID")
                    imdb_character_url = f"https://www.imdb.com/title/tt{imdb_id}/characters/nm{character_id}"
                    imdb_character_urls.append(imdb_character_url)
                    break
            else:
                imdb_character_urls.append("")
        character_introduction = create_character_introduction(imdb_character_names[0], imdb_titles, imdb_id_urls,
                                                               imdb_character_urls)
        for _, crow in character_df.drop_duplicates(["trope", "label"]).iterrows():
            trope, content_text, label = crow["trope"], crow["content-text"], crow["label"]
            trope_url = f"<a href=\"https://tvtropes.org/pmwiki/pmwiki.php/Main/{trope}\" target=\"_blank\">{trope}</a>"
            rows.append([cid, imdb_character_names[0], ";".join(imdb_character_urls), character_introduction, trope,
                         trope_url, content_text, min(imdb_years), label])
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(crowdsource_file, index=False)

if __name__ == '__main__':
    app.run(create_data_file)