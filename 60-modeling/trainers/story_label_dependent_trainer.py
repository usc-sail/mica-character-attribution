"""Train and evaluate the label dependent model on full story"""
import evaluation

import os
import tqdm
import torch
import numpy as np
import pandas as pd
from torch.nn import BCEWithLogitsLoss

IGNORE_VALUE = -100

def get_movie_data(movie, df, tropes, tensors_dir):
    """Get movie data"""
    # get the filepaths
    movie_dir = os.path.join(tensors_dir, movie)
    characters_file = os.path.join(movie_dir, "characters.txt")
    token_ids_file = os.path.join(movie_dir, "token-ids.pt")
    mentions_mask_file = os.path.join(movie_dir, "mentions-mask.pt")
    mention_character_ids_file = os.path.join(movie_dir, "mention-character-ids.pt")
    utterances_mask_file = os.path.join(movie_dir, "utterances-mask.pt")
    names_mask_file = os.path.join(movie_dir, "names-mask.pt")

    # read the imdb character names and convert them to the character ids, for example, "Mark" --> "C098"
    movie_characters = open(characters_file).read().split("\n")
    movie_df = df.loc[df["imdb-id"] == movie, ["imdb-character", "character"]].drop_duplicates()
    movie_df.index = movie_df["imdb-character"]
    movie_characters = [movie_df.loc[movie_character, "character"] for movie_character in movie_characters]

    # read the tensors
    token_ids = torch.load(token_ids_file)
    mentions_mask = torch.load(mentions_mask_file)
    mention_character_ids = torch.load(mention_character_ids_file)
    utterances_mask = torch.load(utterances_mask_file)
    names_mask = torch.load(names_mask_file)

    # get the index of the movie tropes in the full tropes array
    movie_tropes = sorted(df.loc[df["imdb-id"] == movie, "trope"].unique())
    movie_tropes_tindex = torch.LongTensor([tropes.index(movie_trope) for movie_trope in movie_tropes])

    # get the index of the character x trope embeddings for the movie in the character x trope x movie embeddings
    # of the subsuming component
    component_id = df.loc[df["imdb-id"] == movie, "component"].values[0]
    component = df[df["component"] == component_id]
    component_characters = sorted(component["character"].unique())
    component_tropes = sorted(component["trope"].unique())
    component_movies = sorted(component["imdb-id"].unique())
    characters_index = torch.LongTensor([component_characters.index(character) for character in movie_characters])
    movie_tropes_cindex = torch.LongTensor([component_tropes.index(movie_trope) for movie_trope in movie_tropes])
    movie_index = torch.LongTensor([component_movies.index(movie)])
    embeddings_index = torch.meshgrid(characters_index, movie_tropes_cindex, movie_index, indexing="ij")

    return {"token-ids": token_ids,
            "mentions-mask": mentions_mask,
            "mentions-character-ids": mention_character_ids,
            "utterances-mask": utterances_mask,
            "names-mask": names_mask,
            "tropes-index": movie_tropes_tindex,
            "embeddings-index": embeddings_index}

def get_component_data(component, hidden_size, requires_grad):
    """Get component data"""
    component_characters = sorted(component["character"].unique())
    component_tropes = sorted(component["trope"].unique())
    component_movies = sorted(component["imdb-id"].unique())

    # construct the character x trope labels array where value of positive entries (character portrays trope) is 1,
    # value of negative entries (character does not portray trope) is 0, and value of all other entries (we do not
    # know if character portrays or does not portray trope) is IGNORE_VALUE
    component_labels_arr = torch.full((len(component_characters), len(component_tropes)), fill_value=IGNORE_VALUE,
                                        dtype=torch.float32)
    component_character_to_ix = {character:i for i, character in enumerate(component_characters)}
    component_trope_to_ix = {trope:i for i, trope in enumerate(component_tropes)}
    component_labels_df = component.groupby(["character", "trope"]).agg({"label": lambda arr: arr.values[0]})
    for (character, trope), row in component_labels_df.iterrows():
        i, j = component_character_to_ix[character], component_trope_to_ix[trope]
        component_labels_arr[i, j] = row["label"]

    # construct the character x trope labels mask array
    component_labels_mask = (component_labels_arr != IGNORE_VALUE)

    # initialize the component character embeddings matrix
    # it is a zero matrix of shape characters x tropes x movies x hidden size
    component_character_embeddings = torch.zeros(
        (len(component_characters), len(component_tropes), len(component_movies), hidden_size), 
        dtype=torch.float32, requires_grad=requires_grad)

    return {"label": component_labels_arr,
            "mask": component_labels_mask,
            "embeddings": component_character_embeddings}

def model_component(model, classifier, component, component_data, movie_data, trope_token_ids, loss_function):
    """run model and classifier on component"""
    component_movies = sorted(component["imdb-id"].unique())

    # move the component data to gpu
    component_labels_arr = component_data["label"].cuda()
    component_labels_mask = component_data["mask"].cuda()
    component_character_embeddings = component_data["embeddings"].cuda()

    # loop of component movies
    for movie in component_movies:
        # move the movie data to gpu
        movie_token_ids = movie_data[movie]["token-ids"].cuda()
        movie_mentions_mask = movie_data[movie]["mentions-mask"].cuda()
        movie_mentions_character_ids = movie_data[movie]["mentions-character-ids"].cuda()
        movie_utterances_mask = movie_data[movie]["utterances-mask"].cuda()
        movie_names_mask = movie_data[movie]["names-mask"].cuda()

        # get the token ids of the movie tropes
        movie_tropes_index = movie_data[movie]["tropes-index"]
        movie_tropes_token_ids = trope_token_ids[movie_tropes_index].cuda()

        # run model
        movie_character_embeddings = model(movie_token_ids,
                                            movie_names_mask,
                                            movie_utterances_mask,
                                            movie_mentions_mask,
                                            movie_mentions_character_ids,
                                            movie_tropes_token_ids)

        # assign the character embeddings to the component character embeddings
        embeddings_index = movie_data[movie]["embeddings-index"]
        component_character_embeddings[embeddings_index] = movie_character_embeddings.unsqueeze(dim=2)

    # classify whether character portrays trope
    component_character_logits = classifier(component_character_embeddings)
    component_character_logits = torch.max(component_character_logits, dim=2).values

    # compute loss
    component_active_labels = component_labels_arr[component_labels_mask]
    component_active_logits = component_character_logits[component_labels_mask]
    loss = loss_function(component_active_logits, component_active_labels)

    return loss, component_active_logits, component_active_labels

def train(model, classifier, optimizer, train_df, valid_df, tropes, trope_token_ids, tensors_dir, n_epochs):
    train_components = [component for _, component in train_df.groupby("component")]
    valid_components = [component for _, component in valid_df.groupby("component")]
    df = pd.concat([train_df, valid_df], axis=0)
    movie_data = {}
    train_components_data = []
    valid_components_data = []

    for movie in tqdm.tqdm(df["imdb-id"].unique(), desc="reading movie data"):
        movie_data[movie] = get_movie_data(movie, df, tropes, tensors_dir)

    for component in tqdm.tqdm(train_components, desc="setting training component data"):
        train_components_data.append(get_component_data(component, model.hidden_size, True))

    for component in tqdm.tqdm(valid_components, desc="setting validation component data"):
        valid_components_data.append(get_component_data(component, model.hidden_size, False))

    # set up the loss function
    loss_function = BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        # set the model to training mode
        model.train()
        classifier.train()

        # generate a random sequence in which components will be traversed
        components_index = np.random.permutation(len(train_components))

        # initialize the loss, logits, and labels arrays
        train_loss_arr = []
        valid_loss_arr = []
        valid_logits_arr = []
        valid_labels_arr = []

        tbar = tqdm.tqdm(components_index, total=len(train_components), unit="component",
                         desc=f"training epoch {epoch + 1}")
        for component_index in tbar:
            # set the gradients to null
            optimizer.zero_grad()

            # get the component
            component = train_components[component_index]
            component_data = train_components_data[component_index]

            # run model on component
            loss, _, _ = model_component(model, classifier, component, component_data, movie_data, trope_token_ids,
                                         loss_function)

            # backpropagate loss
            loss.backward()

            # update parameters
            optimizer.step()

            # save loss to loss array
            train_loss_arr.append(loss.item())
            tbar.set_description(f"loss = {loss.item():.4f}")

        # average the losses
        avg_train_loss = np.mean(train_loss_arr)
        print(f"epoch {epoch + 1}: train-loss = {avg_train_loss:.4f}")

        # set the model to evaluation mode
        model.eval()
        classifier.eval()

        tbar = tqdm.trange(len(valid_components), unit="component", desc=f"evaluating epoch {epoch + 1}")
        with torch.no_grad():
            for component_index in tbar:
                # get the component
                component = valid_components[component_index]
                component_data = valid_components_data[component_index]

                # run model on component
                loss, logits, labels = model_component(model, classifier, component, component_data, movie_data,
                                                       trope_token_ids, loss_function)

                # save loss to loss array
                valid_loss_arr.append(loss.item())
                tbar.set_description(f"loss = {loss.item():.4f}")

                # save logits and labels
                valid_logits_arr.append(logits)
                valid_labels_arr.append(labels)

        # evaluate
        valid_logits = torch.cat(valid_logits_arr, dim=0)
        valid_labels = torch.cat(valid_labels_arr, dim=0)
        accuracy, precision, recall, f1 = evaluation.evaluate(valid_logits, valid_labels)
        avg_valid_loss = np.mean(valid_loss_arr)
        print(f"epoch {epoch + 1}: valid-loss = {avg_valid_loss:.4f} acc = {accuracy:.1f}, precision = {precision:.1f},"
              f" recall = {recall:.1f}, f1 = {f1:.1f}\n")