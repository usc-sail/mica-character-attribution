import re
import os
import pylcs
import tqdm
import json
import collections
import pandas as pd
from thefuzz import fuzz

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# input/output
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("scripts_dir", default=None, help="scripts directory", required=True)
flags.DEFINE_string("characters", default="10-crawling/movie-linked-characters.csv", 
                    help="csv file of imdb-id, title, year, movie-url, rank, and character-url")
flags.DEFINE_string("tropes", default="10-crawling/movie-character-tropes.csv", 
                    help="csv file of character-url, character, trope, and content-text")
flags.DEFINE_string("imdb_tvtrope_character_map", default="20-preprocessing/imdb-tvtrope-movie-character-map.csv",
                    help="csv file of imdb-id, title, year, imdb-character, character-url, character")
flags.DEFINE_string("bad_imdb_ids_file", default="20-preprocessing/bad-imdb-ids.txt", help="bad imdb ids file")

# parameters
flags.DEFINE_float("min_tok", default=1, help="minimum token set ratio to match imdb character to tvtrope character")
flags.DEFINE_float("min_lcs", default=0.5, help="minimum lcs score to match imdb character to tvtrope")
flags.DEFINE_integer("k", default=10, help=("top k characters to consider from the IMDB cast list in the order they "
                                            "appear"))

def name_match_score(tvtrope_name: str, imdb_name: str) -> tuple[float, float]:
    length = pylcs.lcs_sequence_length(tvtrope_name, imdb_name)
    lcs_score = length/len(tvtrope_name)
    token_set_score = fuzz.token_set_ratio(tvtrope_name, imdb_name)/100
    return lcs_score, token_set_score

def match_characters(tvtrope_characters: list[str], imdb_characters: list[str], threshold_lcs: float, 
                     threshold_tok: float) -> list[tuple[int, int]]:
    lowercased_tvtrope_characters = [x.lower() for x in tvtrope_characters]
    lowercased_imdb_characters = [x.lower() for x in imdb_characters]
    scores = []
    for i, x in enumerate(lowercased_tvtrope_characters):
        for j, y in enumerate(lowercased_imdb_characters):
            score_lcs, score_tok = name_match_score(x, y)
            if score_lcs >= threshold_lcs and score_tok >= threshold_tok:
                scores.append((i, j, (score_lcs, score_tok)))
    scores = sorted(scores, key=lambda item: item[2], reverse=True)
    tvtrope_character_indices = set(range(len(lowercased_tvtrope_characters)))
    imdb_character_indices = set(range(len(lowercased_imdb_characters)))
    matched_indices = []
    for (i, j, (score_lcs, score_tok)) in scores:
        if i in tvtrope_character_indices and j in imdb_character_indices:
            matched_indices.append((i, j, (score_lcs, score_tok)))
            tvtrope_character_indices.remove(i)
            imdb_character_indices.remove(j)
            if not imdb_character_indices:
                break
    return matched_indices

def match_imdb_tvtrope_characters(_):
    # files and directories
    data_dir = FLAGS.data_dir
    movie_characters_file = os.path.join(data_dir, FLAGS.characters)
    character_tropes_file = os.path.join(data_dir, FLAGS.tropes)
    scripts_dir = FLAGS.scripts_dir
    bad_imdb_ids_file = os.path.join(data_dir, FLAGS.bad_imdb_ids_file)
    output_file = os.path.join(data_dir, FLAGS.imdb_tvtrope_character_map)

    # parameters
    threshold_lcs = FLAGS.min_lcs
    threshold_tok = FLAGS.min_tok
    k = FLAGS.k

    # read files
    movie_characters_df = pd.read_csv(movie_characters_file, index_col=None, dtype={"imdb-id": str})
    character_tropes_df = pd.read_csv(character_tropes_file, index_col=None)
    bad_imdb_ids = open(bad_imdb_ids_file).read().strip().split("\n")

    # remove Marvel variant characters
    character_tropes_df = character_tropes_df[~character_tropes_df["character"].str.contains("Variants->")]

    # remove game version characters
    character_tropes_df = character_tropes_df[~character_tropes_df["character"].str.contains("Game version->")]

    characters_df = movie_characters_df.merge(character_tropes_df, how="inner", on="character-url")
    characters_df = characters_df[["imdb-id", "rank", "character-url", "character"]].drop_duplicates()
    imdb_id_to_cast = {}
    imdb_id_to_title = {}
    imdb_id_to_year = {}
    imdb_id_to_genres = {}
    for imdb_id in tqdm.tqdm(os.listdir(scripts_dir), desc="reading cast lists"):
        if imdb_id not in bad_imdb_ids:
            imdb_file = os.path.join(scripts_dir, imdb_id, "imdb.json")
            script_file = os.path.join(scripts_dir, imdb_id, "script.txt")
            if os.path.exists(imdb_file) and os.path.exists(script_file):
                with open(script_file) as fr:
                    script = fr.read().strip()
                n_unique_lines = len(set([line.strip() for line in script.split("\n") if line.strip()]))
                if n_unique_lines >= 100:
                    with open(imdb_file) as fr:
                        imdb_data = json.load(fr)
                    if imdb_id == imdb_data["imdbID"]:
                        cast_list = set()
                        if "cast" in imdb_data:
                            for character in imdb_data["cast"][:k]:
                                if "character" in character:
                                    character_name = character["character"]
                                    if character_name is not None:
                                        cast_list.add(character_name.strip())
                        if cast_list:
                            imdb_id_to_cast[imdb_id] = sorted(cast_list)
                    imdb_id_to_title[imdb_id] = imdb_data["title"]
                    imdb_id_to_year[imdb_id] = imdb_data["year"]
                    imdb_id_to_genres[imdb_id] = ";".join(imdb_data["genres"]) if "genres" in imdb_data else ""
    characters_df = characters_df[characters_df["imdb-id"].isin(imdb_id_to_cast)]
    n_movies = characters_df["imdb-id"].unique().size
    n_characters = characters_df[["character-url", "character"]].drop_duplicates().shape[0]
    print(f"Before matching: {n_movies} movies, {n_characters} characters")

    match_rows = []
    groupby = characters_df.groupby("imdb-id")
    for imdb_id, movie_df in tqdm.tqdm(groupby, total=groupby.ngroups, desc="matching imdb and tvtrope characters"):
        imdb_characters = imdb_id_to_cast[imdb_id]
        title = imdb_id_to_title[imdb_id]
        year = imdb_id_to_year[imdb_id]
        genres = imdb_id_to_genres[imdb_id]
        tvtrope_character_dict = collections.defaultdict(list)
        for _, row in movie_df.iterrows():
            rank, character_url, character_text = row["rank"], row["character-url"], row["character"]
            character_elements = character_text.split("->")
            character_elements = [character_url.split("/")[-1]] + character_elements
            if re.match(r"[#A-Z]-[A-Z]$", character_elements[-1]):
                character_name = character_elements[-2]
            else:
                character_name = character_elements[-1]
            tvtrope_character_dict[character_name].append((rank, character_url, character_text))
        tvtrope_character_names = sorted(list(tvtrope_character_dict.keys()))
        matched_indices = match_characters(tvtrope_character_names, imdb_characters, threshold_lcs, threshold_tok)
        for i, j, (score_lcs, score_tok) in matched_indices:
            tvtrope_character_name = tvtrope_character_names[i]
            imdb_character = imdb_characters[j]
            for rank, character_url, character_text in tvtrope_character_dict[tvtrope_character_name]:
                match_rows.append([imdb_id, title, year, genres, imdb_character, rank, character_url, character_text, 
                                   tvtrope_character_name, score_lcs, score_tok])
    match_df = pd.DataFrame(match_rows,
                            columns=["imdb-id", "title", "year", "genres", "imdb-character", "rank", "character-url",
                                     "character", "character-name", "sequence-score", "token-score"])

    # if an imdb-character is linked to multiple tvtrope-characters
    # and exactly one of them is from the main page (indicated from rank)
    # then remove the other tv-trope characters
    remove_index = []
    for _, df in match_df.groupby(["imdb-id", "imdb-character"]):
        ddf = df[["rank", "character-url", "character"]].drop_duplicates()
        n_main_characters = (ddf["rank"] == "main").sum()
        if n_main_characters == 1:
            index = df[df["rank"] != "main"].index.tolist()
            remove_index.extend(index)
    match_df = match_df[~match_df.index.isin(remove_index)]

    n_movies = match_df["imdb-id"].unique().size
    n_characters = match_df[["imdb-id", "imdb-character"]].drop_duplicates().shape[0]
    print(f"After matching: {n_movies} movies, {n_characters} characters")
    match_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    app.run(match_imdb_tvtrope_characters)