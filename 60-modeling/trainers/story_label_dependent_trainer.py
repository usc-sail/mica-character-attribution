"""Train and evaluate the label dependent model on full story"""
import os
import tqdm
import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss

IGNORE_VALUE = -100

def train(model, classifier, optimizer, df, tropes, trope_token_ids, tensors_dir, n_epochs):
    components = [component for _, component in df.groupby("component")]

    movie_data = {}
    for movie in tqdm.tqdm(df["imdb-id"].unique(), desc="reading movie data"):
        movie_dir = os.path.join(tensors_dir, movie)
        characters_file = os.path.join(movie_dir, "characters.txt")
        token_ids_file = os.path.join(movie_dir, "token-ids.pt")
        mentions_mask_file = os.path.join(movie_dir, "mentions-mask.pt")
        mention_character_ids_file = os.path.join(movie_dir, "mention-character-ids.pt")
        utterances_mask_file = os.path.join(movie_dir, "utterances-mask.pt")
        names_mask_file = os.path.join(movie_dir, "names-mask.pt")
        movie_characters = open(characters_file).read().split("\n")
        movie_df = df.loc[df["imdb-id"] == movie, ["imdb-character", "character"]].drop_duplicates()
        movie_df.index = movie_df["imdb-character"]
        movie_characters = [movie_df.loc[movie_character, "character"] for movie_character in movie_characters]
        token_ids = torch.load(token_ids_file)
        mentions_mask = torch.load(mentions_mask_file)
        mention_character_ids = torch.load(mention_character_ids_file)
        utterances_mask = torch.load(utterances_mask_file)
        names_mask = torch.load(names_mask_file)
        movie_tropes = sorted(df.loc[df["imdb-id"] == movie, "trope"].unique())
        movie_tropes_tindex = torch.LongTensor([tropes.index(movie_trope) for movie_trope in movie_tropes])
        component_id = df.loc[df["imdb-id"] == movie, "component"].values[0]
        component = df[df["component"] == component_id]
        component_characters = sorted(component["character"].unique())
        component_tropes = sorted(component["trope"].unique())
        component_movies = sorted(component["imdb-id"].unique())
        characters_index = torch.LongTensor([component_characters.index(character) for character in movie_characters])
        movie_tropes_cindex = torch.LongTensor([component_tropes.index(movie_trope) for movie_trope in movie_tropes])
        movie_index = torch.LongTensor([component_movies.index(movie)])
        embeddings_index = torch.meshgrid(characters_index, movie_tropes_cindex, movie_index, indexing="ij")
        movie_data[movie] = {"token-ids": token_ids,
                             "mentions-mask": mentions_mask,
                             "mentions-character-ids": mention_character_ids,
                             "utterances-mask": utterances_mask,
                             "names-mask": names_mask,
                             "tropes-index": movie_tropes_tindex,
                             "embeddings-index": embeddings_index}

    components_data = []
    for component in tqdm.tqdm(components, desc="setting component data"):
        component_characters = sorted(component["character"].unique())
        component_tropes = sorted(component["trope"].unique())
        component_movies = sorted(component["imdb-id"].unique())
        component_labels_arr = torch.full((len(component_characters), len(component_tropes)), fill_value=IGNORE_VALUE)
        component_character_to_ix = {character:i for i, character in enumerate(component_characters)}
        component_trope_to_ix = {trope:i for i, trope in enumerate(component_tropes)}
        component_labels_df = component.groupby(["character", "trope"]).agg({"label": lambda arr: arr.values[0]})
        for (character, trope), row in component_labels_df.iterrows():
            i, j = component_character_to_ix[character], component_trope_to_ix[trope]
            component_labels_arr[i, j] = row["label"]
        component_labels_mask = (component_labels_arr != IGNORE_VALUE)
        component_character_embeddings = torch.zeros(
            (len(component_characters), len(component_tropes), len(component_movies), model.hidden_size), 
            dtype=float, requires_grad=True)
        components_data.append({"label": component_labels_arr, "mask": component_labels_mask,
                                "embeddings": component_character_embeddings})

    loss_function = BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        model.train()
        classifier.train()
        optimizer.zero_grad()
        components_index = np.random.permutation(len(components))
        loss_arr = []
        tbar = tqdm.tqdm(enumerate(components_index), total=len(components))
        for i, component_index in tbar:
            torch.cuda.empty_cache()
            component = components[component_index]
            component_data = components_data[component_index]
            component_movies = sorted(component["imdb-id"].unique())
            component_labels_arr = component_data["label"].cuda()
            component_labels_mask = component_data["mask"].cuda()
            component_character_embeddings = component_data["embeddings"].cuda()
            for j, movie in enumerate(component_movies):
                movie_token_ids = movie_data[movie]["token-ids"].cuda()
                movie_mentions_mask = movie_data[movie]["mentions-mask"].cuda()
                movie_mentions_character_ids = movie_data[movie]["mentions-character-ids"].cuda()
                movie_utterances_mask = movie_data[movie]["utterances-mask"].cuda()
                movie_names_mask = movie_data[movie]["names-mask"].cuda()
                movie_tropes_index = movie_data[movie]["tropes-index"]
                movie_tropes_token_ids = trope_token_ids[movie_tropes_index].cuda()
                movie_character_embeddings = model(movie_token_ids,
                                                    movie_names_mask,
                                                    movie_utterances_mask,
                                                    movie_mentions_mask,
                                                    movie_mentions_character_ids,
                                                    movie_tropes_token_ids)
                embeddings_index = movie_data[movie]["embeddings-index"].cuda()
                component_character_embeddings[embeddings_index] = movie_character_embeddings
            component_character_logits = classifier(component_character_embeddings)
            component_character_logits = torch.max(component_character_logits, dim=2)
            component_active_labels = component_labels_arr[component_labels_mask]
            component_active_logits = component_character_logits[component_labels_mask]
            loss = loss_function(component_active_logits, component_active_labels)
            loss.backward()
            loss_arr.append(loss.item())
            tbar.set_description(f"loss = {loss.item():.4f}")
        optimizer.step()
        avg_loss = np.mean(loss_arr)
        print(f"epoch = {epoch + 1}: loss = {avg_loss:.4f}")