"""Partition the data into train, dev, and test.
This is done by first creating a bipartite graph of characters and movies.
A character vertex is adjacent to a movie vertex if the character appears in the movie.
Next, we find the connected components of this graph, and partition them into train, dev, and test sets.
"""
import re
import os
import pandas as pd
import numpy as np
import random
import tqdm
import collections
from scipy.special import rel_entr

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("character_map", default="20-preprocessing/imdb-tvtrope-movie-character-map.csv",
                    help="csv file of imdb-id, title, year, imdb-character, character-url, character, character-name")
flags.DEFINE_string("tropes", default="10-crawling/movie-character-tropes.csv", help="csv file of "
                    "character-url, character, trope, content-text")
flags.DEFINE_string("output", default="20-preprocessing/train-dev-test-splits.csv", help="csv file of partition, "
                    "imdb-id, character, imdb-character, tvtrope-character-url, tvtrope-character-xpath, trope, "
                    "content-text")
flags.DEFINE_integer("k", default=5000, help="top k tropes to keep")
flags.DEFINE_integer("q", default=100, help="top q tropes to use to compute kl divergence")
flags.DEFINE_integer("trn", default=400, help="number of train components")
flags.DEFINE_integer("dev", default=90, help="number of dev components")

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

def unique_sort(components):
    """uniquely arrange components"""
    identifiers = []
    for _, movies in components:
        identifier = sorted(movies)[0]
        identifiers.append(identifier)
    sort_index = np.argsort(identifiers)
    components = [components[i] for i in sort_index]
    return components

def get_trope_distribution(components, character_to_tropes, tropes):
    component_tropes = []
    for component in components:
        for character in component[0]:
            component_tropes.extend(character_to_tropes[character])
    distribution = collections.Counter(component_tropes)
    distribution = collections.defaultdict(int, dict(distribution))
    distribution = [distribution[trope]/len(component_tropes) for trope in tropes]
    return distribution

def split_train_dev_test(_):
    # read command-line flags
    data_dir = FLAGS.data_dir
    map_file = os.path.join(data_dir, FLAGS.character_map)
    tropes_file = os.path.join(data_dir, FLAGS.tropes)
    output_file = os.path.join(data_dir, FLAGS.output)
    k = FLAGS.k
    q = FLAGS.q
    trn = FLAGS.trn
    dev = FLAGS.dev

    # read imdb to tvtrope map
    # read character tropes
    map_df = pd.read_csv(map_file, index_col=None, dtype={"imdb-id": str})
    tropes_df = pd.read_csv(tropes_file, index_col=None)

    # join imdb to tvtrope map with character tropes dataframe
    characters_df = map_df.merge(tropes_df, how="inner", on=["character-url", "character"])
    characters_df["trope"] = characters_df["trope"].str.split("/").str[-1]
    characters_df.rename(columns={"character-url": "tvtrope-character-url", "character": "tvtrope-character-xpath",
                                  "character-name": "tvtrope-character-name"}, 
                         inplace=True)
    characters_df["tvtrope-character-xpath"] = characters_df["tvtrope-character-xpath"].apply(
        lambda x: re.sub(r"(^|(->))[#A-Z]-[A-Z]$", "", x).strip())

    # find tropes to characters mapping
    # tropes is the tuple of tropes in sorted order
    tropes_to_characters = collections.defaultdict(set)
    for character, df in characters_df.groupby(["tvtrope-character-url", "tvtrope-character-xpath"]):
        assert len(df["tvtrope-character-name"].unique()) == 1
        tvtrope_character_name = df["tvtrope-character-name"].values[0]
        tropes = (tuple(sorted(df["trope"].unique().tolist())), tvtrope_character_name)
        tropes_to_characters[tropes].add(character)
    tropes_to_characters = dict(tropes_to_characters)

    # find characters to character id mapping
    character_to_characterid = {}
    w = int(np.log10(len(tropes_to_characters))) + 1
    for i, (tropes, characters) in enumerate(tropes_to_characters.items()):
        characterid = "C" + f"{i}".zfill(w)
        for character in characters:
            character_to_characterid[character] = characterid

    # create the character column
    character_col = []
    for _, row in characters_df.iterrows():
        character_col.append(character_to_characterid[(row["tvtrope-character-url"], row["tvtrope-character-xpath"])])
    characters_df["character"] = character_col
    characters_df = characters_df[["imdb-id", "title", "year", "genres", "character", "imdb-character", "rank",
                                   "tvtrope-character-url", "tvtrope-character-xpath", "tvtrope-character-name",
                                   "sequence-score", "token-score", "trope", "content-text"]]

    # remove imdb-characters that are mapped to multiple characters
    remove_index = set()
    for _, df in characters_df.groupby(["imdb-id", "imdb-character"]):
        if df["character"].unique().size > 1:
            remove_index.update(df.index.tolist())
    characters_df = characters_df[~characters_df.index.isin(remove_index)]

    # retain top k tropes
    tropes = characters_df["trope"].tolist()
    tropes_dict = collections.Counter(tropes)
    tropes = sorted(tropes_dict.keys(), key=lambda trope: tropes_dict[trope], reverse=True)
    selected_tropes = tropes[:k]
    percentage = 100*characters_df["trope"].isin(selected_tropes).sum()/len(characters_df)
    minimum = tropes_dict[selected_tropes[-1]]
    print(f"top {k} tropes are selected")
    print(f"covers {percentage:.2f}% data, each trope portrayed by at least {minimum} characters")
    characters_df = characters_df[characters_df["trope"].isin(selected_tropes)]
    characters_df = characters_df.drop_duplicates(subset=["imdb-id", "character", "trope"])
    n_movies = characters_df["imdb-id"].unique().size
    n_characters = characters_df["character"].unique().size
    n_tropes = characters_df["trope"].unique().size
    print(f"{n_movies} movies, {n_characters} characters, {n_tropes} tropes, "
          f"{len(characters_df)} samples (movie x character x tropes)\n")

    # find character to movies, character to tropes mapping
    # find movie to characters mapping
    character_to_movies = {}
    character_to_tropes = {}
    movie_to_characters = {}
    for character, df in characters_df.groupby("character"):
        character_to_movies[character] = set(df["imdb-id"].tolist())
        character_to_tropes[character] = set(df["trope"].tolist())
    for movie, df in characters_df.groupby("imdb-id"):
        movie_to_characters[movie] = set(df["character"].tolist())
    components = find_connected_components(character_to_movies, movie_to_characters)
    components = unique_sort(components)
    print(f"{len(components)} components")

    # create the component column
    character_to_componentid = {}
    w = int(np.log10(len(components))) + 1
    for i, component in enumerate(components):
        for character in component[0]:
            character_to_componentid[character] = "N" + f"{i}".zfill(w)
    component_col = []
    for _, row in characters_df.iterrows():
        component_col.append(character_to_componentid[row["character"]])
    characters_df["component"] = component_col

    # find the partition of components with the minimum kl divergence in trope distribution
    tst = len(components) - trn - dev
    print(f"{trn} train, {dev} dev, and {tst} test components")
    best_split, min_divergence = None, float("inf")
    random.seed(2024)
    for _ in tqdm.trange(100, desc="finding best split"):
        index = list(range(len(components)))
        random.shuffle(index)
        trn_components = [components[i] for i in index[:trn]]
        dev_components = [components[i] for i in index[trn:trn+dev]]
        tst_components = [components[i] for i in index[trn+dev:]]
        trn_dist = get_trope_distribution(trn_components, character_to_tropes, selected_tropes)
        dev_dist = get_trope_distribution(dev_components, character_to_tropes, selected_tropes)
        tst_dist = get_trope_distribution(tst_components, character_to_tropes, selected_tropes)
        divergence = rel_entr(dev_dist, trn_dist)[:q].sum() + rel_entr(tst_dist, trn_dist)[:q].sum()
        if divergence < min_divergence:
            best_split = index
            min_divergence = divergence
    print(f"minimum divergence = {min_divergence:.2e}")
    print()

    # create the partition column
    characters_df["partition"] = ""
    trn_components = [components[i] for i in best_split[:trn]]
    dev_components = [components[i] for i in best_split[trn:trn+dev]]
    tst_components = [components[i] for i in best_split[trn+dev:]]
    trn_characters = [character for component in trn_components for character in component[0]]
    dev_characters = [character for component in dev_components for character in component[0]]
    tst_characters = [character for component in tst_components for character in component[0]]
    characters_df.loc[characters_df["character"].isin(trn_characters), "partition"] = "train"
    characters_df.loc[characters_df["character"].isin(dev_characters), "partition"] = "dev"
    characters_df.loc[characters_df["character"].isin(tst_characters), "partition"] = "test"
    assert all(characters_df["partition"] != "")
    total_n_movies = characters_df["imdb-id"].unique().size
    total_n_characters = characters_df["character"].unique().size
    total_n_tropes = characters_df["trope"].unique().size
    total_n_samples = len(characters_df)
    for partition in ["train", "dev", "test"]:
        n_movies = characters_df.loc[characters_df["partition"] == partition, "imdb-id"].unique().size
        n_characters = characters_df.loc[characters_df["partition"] == partition, "character"].unique().size
        n_tropes = characters_df.loc[characters_df["partition"] == partition, "trope"].unique().size
        n_samples = (characters_df["partition"] == partition).sum()
        percentage_movies = 100 * n_movies / total_n_movies
        percentage_characters = 100 * n_characters / total_n_characters
        percentage_tropes = 100 * n_tropes / total_n_tropes
        percentage_samples = 100 * n_samples / total_n_samples
        print(f"{partition:5s}: {n_movies} movies ({percentage_movies:.1f}%), {n_characters} characters "
              f"({percentage_characters:.1f}%), {n_tropes} tropes ({percentage_tropes:.1f}%), {n_samples} samples "
              f"({percentage_samples:.1f}%)")

    # write to output
    characters_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    app.run(split_train_dev_test)