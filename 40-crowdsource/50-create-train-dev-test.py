"""Create the train, dev and test sets for CHATTER"""
import collections
import os
import pandas as pd
import re

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("datadir", default=None, help="data directory", required=True)

def find_connected_components(character_to_movies, movie_to_characters):
    """Use BFS algorithm to find connected components"""
    characters = set(character_to_movies.keys())
    movies = set(movie_to_characters.keys())
    components = []

    while characters:
        component_characters = set()
        component_movies = set()
        character = characters.pop()
        queue = [character]
        while queue:
            top = queue[0]
            ismovie = re.match(r"\d+$", top) is not None
            if ismovie:
                for character in movie_to_characters[top]:
                    if character not in queue and character not in component_characters:
                        queue.append(character)
                movies.discard(top)
                component_movies.add(top)
            else:
                for movie in character_to_movies[top]:
                    if movie not in queue and movie not in component_movies:
                        queue.append(movie)
                characters.discard(top)
                component_characters.add(top)
            queue = queue[1:]
        components.append((component_characters, component_movies))

    assert not characters and not movies
    return components

def create_chatter(_):
    data_dir = FLAGS.datadir
    data_file = os.path.join(data_dir, "60-modeling/dataset-with-only-character-tropes.csv")
    test_file = os.path.join(data_dir, "60-modeling/test-dataset.csv")
    tropes_file = os.path.join(data_dir, "60-modeling/character-tropes.csv")

    data_df = pd.read_csv(data_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
    test_df = pd.read_csv(test_file, index_col=None, dtype={"combination": str})
    tropes_df = pd.read_csv(tropes_file, index_col=None)

    character2movies = collections.defaultdict(set)
    movie2characters = collections.defaultdict(set)
    for character, movie in data_df[["character", "imdb-id"]].drop_duplicates().itertuples(index=False, name=None):
        character2movies[character].add(movie)
        movie2characters[movie].add(character)
    character2movies = dict(character2movies)
    movie2characters = dict(movie2characters)

    test_characters = set(test_df["character"])
    testdev_movies = set([movie for character in test_characters for movie in character2movies[character]])
    characters_in_testdev_movies = set([character for movie in testdev_movies for character in movie2characters[movie]])
    testdev_characters = set([character for character in characters_in_testdev_movies
                              if character2movies[character].issubset(testdev_movies)])
    train_characters = set(character2movies.keys()).difference(characters_in_testdev_movies)
    train_movies = set([movie for character in train_characters for movie in character2movies[character]])
    assert all([movie not in train_movies for character in testdev_characters for movie in character2movies[character]])
    assert all([movie not in testdev_movies for character in train_characters for movie in character2movies[character]])

    charactermoviemap_df = (data_df[["character", "imdb-id", "imdb-character"]].drop_duplicates()
                            .rename(columns={"imdb-character": "name"}))
    moviemetadata_df = data_df[["imdb-id", "title", "year", "genres"]].drop_duplicates()
    chatter_df = (data_df[["character", "trope", "label", "content-text"]].drop_duplicates()
                  .rename(columns={"label": "tvtrope-label", "content-text": "tvtrope-explanation"}))
    test_df["label"] = (test_df["conf"] > 0).astype(int)
    test_df["weight"] = (test_df["conf"].abs() - 2)/4
    chatter_df = chatter_df.merge(test_df[["character", "trope", "label", "weight"]],
                                  how="left", on=["character", "trope"])
    chatter_df["partition"] = ""
    chatter_df.loc[chatter_df["character"].isin(train_characters), "partition"] = "train"
    chatter_df.loc[chatter_df["character"].isin(testdev_characters), "partition"] = "dev"
    chatter_df.loc[chatter_df["label"].notna(), "partition"] = "test"

    for partition in ["train", "dev", "test"]:
        partition_df = chatter_df[chatter_df["partition"] == partition]
        nsamples = len(partition_df)
        ncharacters = len(partition_df["character"].unique())
        ntropes = len(partition_df["trope"].unique())
        nmovies = len(charactermoviemap_df.loc[charactermoviemap_df["character"].isin(partition_df["character"]),
                                               "imdb-id"].unique())
        print(f"{partition:5s}: {nsamples:5d} samples, {ncharacters:4d} characters, {ntropes:5d} tropes, "
              f"{nmovies:3d} movies")

    tropes_df = tropes_df[["trope", "definition", "summary"]].drop_duplicates()

    components = find_connected_components(character2movies, movie2characters)
    charactermoviemap_df["component"] = ""
    for i, (_, movies) in enumerate(components):
        charactermoviemap_df.loc[charactermoviemap_df["imdb-id"].isin(movies), "component"] = "N" + f"{i}".zfill(3)

    charactermoviemap_file = os.path.join(data_dir, "CHATTER/character-movie-map.csv")
    moviemetadata_file = os.path.join(data_dir, "CHATTER/movie-metadata.csv")
    tropes_file = os.path.join(data_dir, "CHATTER/tropes.csv")
    chatter_file = os.path.join(data_dir, "CHATTER/chatter.csv")
    charactermoviemap_df.to_csv(charactermoviemap_file, index=False)
    moviemetadata_df.to_csv(moviemetadata_file, index=False)
    tropes_df.to_csv(tropes_file, index=False)
    chatter_df.to_csv(chatter_file, index=False)

if __name__ == '__main__':
    app.run(create_chatter)