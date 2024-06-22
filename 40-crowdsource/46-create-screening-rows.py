"""Create csv file containing fields for IMDb id, title, url, and picture url. This file will be used for the 
screening task"""
import os
import re
import json
import collections
import pandas as pd

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("movie_scripts_dir", default="movie-scripts", help="movie scripts directory")
flags.DEFINE_string("crowdsource_file", default="40-crowdsource/dataset-to-annotate.csv",
                    help="csv file used for the annotation task")
flags.DEFINE_string("screening_file", default="40-crowdsource/screening-movies.csv",
                    help="csv file to be used for the screening task")

def extract_imdb_ids(character_urls):
    imdb_ids = []
    for url in character_urls.split(";"):
        match = re.search(r"title/tt(\d+)", url)
        imdb_ids.append(match.group(1))
    return ";".join(imdb_ids)

def create_screening_rows(_):
    data_dir = FLAGS.data_dir
    movie_scripts_dir = os.path.join(data_dir, FLAGS.movie_scripts_dir)
    crowdsource_file = os.path.join(data_dir, FLAGS.crowdsource_file)
    screening_file = os.path.join(data_dir, FLAGS.screening_file)

    crowdsource_df = pd.read_csv(crowdsource_file, index_col=None)
    crowdsource_df["imdb-ids"] = crowdsource_df["character-urls"].apply(extract_imdb_ids)
    samples = set(crowdsource_df.index.tolist())
    imdb_id_to_samples = collections.defaultdict(set)
    for index, row in crowdsource_df.iterrows():
        for imdb_id in row["imdb-ids"].split(";"):
            imdb_id_to_samples[imdb_id].add(index)

    greedy_movies = []
    greedy_coverage = []
    uncovered_samples = set(crowdsource_df.index.tolist())
    while len(uncovered_samples) > 0:
        greedy_movie = None
        max_n = 0
        for imdb_id, imdb_samples in imdb_id_to_samples.items():
            if imdb_id not in greedy_movies:
                n = len(uncovered_samples.intersection(imdb_samples))
                if greedy_movie is not None:
                    if n > max_n:
                        max_n = n
                        greedy_movie = imdb_id
                else:
                    greedy_movie = imdb_id
                    max_n = n
        assert greedy_movie is not None and greedy_movie not in greedy_movies
        greedy_movies.append(greedy_movie)
        uncovered_samples.difference_update(imdb_id_to_samples[greedy_movie])
        greedy_coverage.append(100 * (len(samples) - len(uncovered_samples))/len(samples))

    rows = []
    for imdb_id, coverage in zip(greedy_movies, greedy_coverage):
        imdb_file = os.path.join(movie_scripts_dir, imdb_id, "imdb.json")
        with open(imdb_file) as fr:
            imdb_data = json.load(fr)
        picture_url = imdb_data["cover url"]
        title = imdb_data["title"]
        year = imdb_data["year"]
        genres = ";".join(imdb_data["genres"])
        rows.append([imdb_id, title, year, genres, picture_url, coverage])
    screening_df = pd.DataFrame(rows, columns=["imdb-id", "title", "year", "genres", "picture-url", "coverage"])
    screening_df.to_csv(screening_file, index=False)

if __name__ == '__main__':
    app.run(create_screening_rows)