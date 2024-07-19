"""Train and evaluate the label dependent model on character segments"""
import os
import math
import tqdm
import torch
import random
import numpy as np
from torch.nn import BCEWithLogitsLoss

def train(model, classifier, optimizer, df, tropes, trope_token_ids, tensors_dir, n_epochs, ncharacters_batch):
    character_data = {}

    # read the character tensors
    for character, character_df in tqdm.tqdm(df.groupby("character"), desc="reading character data",
                                             total=df["character"].unique().size):
        # initialize the arrays to save the character tensors
        token_ids_arr, mentions_mask_arr, utterances_mask_arr, names_mask_arr = [], [], [], []
        mention_character_ids_arr = []

        # iterate over movies where the character has appeared
        for _, row in character_df.drop_duplicates("imdb-id").iterrows():
            movie, imdb_character = row["imdb-id"], row["imdb-character"]
            character_tensors_dir = os.path.join(tensors_dir, movie, imdb_character)

            if os.path.exists(character_tensors_dir):
                # read the tensors if the tensors directory exists
                # tensors directory exists if the character has at least some mention or utterance
                token_ids = torch.load(os.path.join(character_tensors_dir, "token-ids.pt"))
                mentions_mask = torch.load(os.path.join(character_tensors_dir, "mentions-mask.pt"))
                utterances_mask = torch.load(os.path.join(character_tensors_dir, "utterances-mask.pt"))
                names_mask = torch.load(os.path.join(character_tensors_dir, "names-mask.pt"))
                mention_character_ids = torch.zeros(len(mentions_mask), dtype=torch.int64)

                # add dimension
                utterances_mask = utterances_mask.unsqueeze(dim=0)
                names_mask = names_mask.unsqueeze(dim=0)

                # save tensors to arrays
                token_ids_arr.append(token_ids)
                mentions_mask_arr.append(mentions_mask)
                utterances_mask_arr.append(utterances_mask)
                names_mask_arr.append(names_mask)
                mention_character_ids_arr.append(mention_character_ids)

        if token_ids_arr:
            # add character data if the arrays are not empty
            character_tropes, character_trope_labels = [], []
            for _, row in character_df.drop_duplicates("trope").iterrows():
                character_tropes.append(row["trope"])
                character_trope_labels.append(row["label"])
            character_tropes_index = torch.LongTensor([tropes.index(character_trope)
                                                       for character_trope in character_tropes])
            character_trope_labels = torch.FloatTensor(character_trope_labels)
            character_data[character] = {"token-ids": token_ids_arr,
                                         "mentions-mask": mentions_mask_arr,
                                         "utterances-mask": utterances_mask_arr,
                                         "names-mask": names_mask_arr,
                                         "mention-character-ids": mention_character_ids_arr,
                                         "tropes-index": character_tropes_index,
                                         "label": character_trope_labels}

    # set the models to training mode
    model.train()
    classifier.train()

    # calculate number of batches
    nbatches = math.ceil(len(character_data)/ncharacters_batch)

    # set up the loss function
    loss_function = BCEWithLogitsLoss()

    # outer loop of epochs
    for epoch in range(n_epochs):
        characters = sorted(character_data.keys())
        random.shuffle(characters)
        batch_loss_arr = []

        # middle loop of batches
        tbar = tqdm.trange(nbatches, unit="batch characters")
        for batch in tbar:
            batch_characters = characters[batch * ncharacters_batch: (batch + 1) * ncharacters_batch]

            # set the gradients to null
            optimizer.zero_grad()
            loss_arr = []

            # second middle loop of characters in the batch
            for character in batch_characters:
                data = character_data[character]
                tropes_index = data["tropes-index"]
                character_trope_token_ids = trope_token_ids[tropes_index].cuda()
                labels = data["label"].cuda()
                character_embeddings_arr = []
                n = len(data["token-ids"])

                # inner loop of movies where character has appeared
                for i in range(n):
                    token_ids = data["token-ids"][i].cuda()
                    mentions_mask = data["mentions-mask"][i].cuda()
                    utterances_mask = data["utterances-mask"][i].cuda()
                    names_mask = data["names-mask"][i].cuda()
                    mention_character_ids = data["mention-character-ids"][i].cuda()

                    # run model
                    movie_character_embeddings = model(token_ids,
                                                       names_mask,
                                                       utterances_mask,
                                                       mentions_mask,
                                                       mention_character_ids,
                                                       character_trope_token_ids)

                    # save the embeddings
                    character_embeddings_arr.append(movie_character_embeddings)

                # concatenate the character embeddings from each movie
                character_embeddings = torch.cat(character_embeddings_arr, dim=0)

                # classify whether character portrays trope
                logits = classifier(character_embeddings)
                logits = torch.max(logits, dim=0).values

                # calculate loss
                loss = loss_function(logits, labels)
                loss_arr.append(loss.reshape(1))

            # calculate batch loss
            loss_arr = torch.cat(loss_arr)
            batch_loss = torch.mean(loss_arr)

            # backpropagate loss
            batch_loss.backward()

            # update parameters
            optimizer.step()

            # save batch loss to array
            batch_loss_arr.append(batch_loss.item())
            tbar.set_description(f"loss = {batch_loss.item():.4f}")

        # average the batch losses
        avg_batch_loss = np.mean(batch_loss_arr)
        print(f"epoch = {epoch + 1}: loss = {avg_batch_loss:.4f}")