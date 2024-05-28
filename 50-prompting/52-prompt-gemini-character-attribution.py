"""Prompt gemini-1.5-pro to predict whether given a complete movie script, a character name, and a trope, classify
whether the character portrays the trope anywhere in the story"""
import os
import time
import tqdm
import pandas as pd
from google import generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("gemini_key_file", default=None, required=True, help="file containing Gemini API key")
flags.DEFINE_integer("k", default=None, required=True, lower_bound=0, help="number of character-trope pairs to prompt")
flags.DEFINE_string("prompt_template_file", default=None, required=True, help="prompt template text file")
flags.DEFINE_string("prompt_response_file", default=None, required=True,
                    help="csv file to output the result of prompting full movie scripts for character tropes")
flags.DEFINE_string("list_characters_file", default="50-prompting/prompt-list-characters.csv",
                    help="csv file containing character lists found from prompting complete movie scripts")
flags.DEFINE_string("dataset_file", default="60-modeling/dataset.csv", help="dataset csv file")
flags.DEFINE_string("movie_scripts_dir", default="movie-scripts", help="movie scripts directory")
flags.DEFINE_string("tropes_file", default="60-modeling/tropes.csv", help="csv file containing trope definitions")

def uniq_size(arr):
    return arr.unique().size

def first_value(arr):
    return arr.values[0]

def prompt_character_attrs(_):
    data_dir = FLAGS.data_dir
    gemini_key_file = FLAGS.gemini_key_file
    n_samples = FLAGS.k
    prompt_template_file = os.path.join(data_dir, FLAGS.prompt_template_file)
    dataset_file = os.path.join(data_dir, FLAGS.dataset_file)
    tropes_file = os.path.join(data_dir, FLAGS.tropes_file)
    movie_scripts_dir = os.path.join(data_dir, FLAGS.movie_scripts_dir)
    characters_list_file = os.path.join(data_dir, FLAGS.list_characters_file)
    prompt_response_file = os.path.join(data_dir, FLAGS.prompt_response_file)

    with open(gemini_key_file) as fr:
        gemini_api_key = fr.read().strip()

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    dataset_df = pd.read_csv(dataset_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    characters_list_df = pd.read_csv(characters_list_file, index_col=None, dtype={"imdb-id": str})

    non_blocking_imdb_ids = set(characters_list_df.loc[characters_list_df["response"].notna(), "imdb-id"].tolist())
    character_movies_df = (dataset_df
                           .groupby("character")
                           .agg({"imdb-id": [uniq_size, first_value], "imdb-character": first_value})
                           .reset_index())
    character_movies_df.columns = ["character", "n-movies", "imdb-id", "imdb-character"]
    character_movies_df = character_movies_df[character_movies_df["n-movies"] == 1]
    dataset_df = dataset_df[dataset_df["imdb-id"].isin(non_blocking_imdb_ids) 
                            & dataset_df["character"].isin(character_movies_df["character"])]
    dataset_df.sort_values(by=["character", "trope"], inplace=True)
    sample_df = dataset_df.sample(n=n_samples, random_state=0)[["character", "trope", "content-text", "label"]]
    sample_df = sample_df.merge(character_movies_df, on="character", how="left")
    sample_df = sample_df.merge(tropes_df, on="trope", how="left")
    sample_df = sample_df[["character", "imdb-character", "imdb-id", "trope", "definition", "content-text", "label"]]

    imdb_id_to_script = {}
    for imdb_id in sample_df["imdb-id"].unique():
        script_file = os.path.join(movie_scripts_dir, imdb_id, "script.txt")
        imdb_id_to_script[imdb_id] = open(script_file).read().strip()

    prompt_template = open(prompt_template_file).read().strip()
    safety_settings={HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE}
    responses = []
    for _, row in tqdm.tqdm(sample_df.iterrows(), total=len(sample_df), unit="sample",
                            desc="prompting for attribution"):
        script = imdb_id_to_script[row["imdb-id"]]
        prompt = (prompt_template
                  .replace("$$SCRIPT$$", script)
                  .replace("$$DEFINITION$$", row["definition"])
                  .replace("$$CHARACTER$$", row["imdb-character"])
                  .replace("$$TROPE$$", row["trope"]))
        response = model.generate_content(prompt, safety_settings=safety_settings)
        try:
            response_text = response.text
            responses.append(response_text)
        except Exception:
            error_message = f"ERROR: prompt-feedback = {response.prompt_feedback}"
            if len(response.candidates):
                error_message += f" finish-reason = {response.candidates[0].finish_reason}"
                error_message += f" safety-ratings = {response.candidates[0].safety_ratings}"
            responses.append(error_message)
        time.sleep(30)
    sample_df["response"] = responses
    sample_df.to_csv(prompt_response_file, index=False)

if __name__ == '__main__':
    app.run(prompt_character_attrs)