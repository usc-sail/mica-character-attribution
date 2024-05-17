"""Prompt gemini-1.5-pro to find the list of characters in the movie script"""
import os
import time
import pandas as pd
from google import generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, required=True, help="data directory")
flags.DEFINE_string("gemini_key_file", default=None, required=True, help="file containing Gemini API key")
flags.DEFINE_string("dataset_file", default="40-modeling/dataset.csv", help="dataset csv file")
flags.DEFINE_string("movie_scripts_dir", default="movie-scripts", help="movie scripts directory")
flags.DEFINE_string("response_file", default="50-prompting/prompt-list-characters.txt",
                    help="output csv file where the list of characters is written")

def list_characters(_):
    data_dir = FLAGS.data_dir
    gemini_key_file = FLAGS.gemini_key_file
    dataset_file = os.path.join(data_dir, FLAGS.dataset_file)
    movie_scripts_dir = os.path.join(data_dir, FLAGS.movie_scripts_dir)
    response_file = os.path.join(data_dir, FLAGS.response_file)

    with open(gemini_key_file) as fr:
        gemini_api_key = fr.read().strip()

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    dataset_df = pd.read_csv(dataset_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
    imdb_ids = dataset_df["imdb-id"].unique()
    rows = []

    for imdb_id in imdb_ids:
        script_file = os.path.join(movie_scripts_dir, imdb_id, "script.txt")
        script = open(script_file).read().strip()
        prompt = f"List the characters mentioned in the given movie script.\n\n{script}"
        response = model.generate_content(
            prompt,
            safety_settings={HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE})
        print(f"imdb-id = {imdb_id}")
        try:
            response_text = response.text
            rows.append([imdb_id, response_text])
            print("successful!")
        except Exception:
            rows.append([imdb_id, ""])
            print(response.prompt_feedback)
            print(response.candidates[0].finish_reason)
            print(response.candidates[0].safety_ratings)
        print("\n\n")
        time.sleep(30)

    response_df = pd.DataFrame(rows, columns=["imdb-id", "response"])
    response_df.to_csv(response_file, index=False)

if __name__ == '__main__':
    app.run(list_characters)