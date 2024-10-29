"""Train and evaluate the label dependent model on full story"""
import evaluation
from dataloaders.data import Chatter

import tqdm
import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss

def get_component_tensors(chatter: Chatter, imdbid_to_tensors, alltropes, hidden_size):
    """Get component data"""
    componentid_to_tensors = {}
    componentids = chatter.train_componentids + chatter.dev_componentids
    for i, componentid in tqdm.tqdm(enumerate(componentids), desc="creating batch tensors", unit="batch",
                                    total=len(componentids)):
        characterids = set()
        tropes = set()
        imdbids = set()
        for characterid, trope in chatter.componentid_to_characterids_and_tropes[componentid]:
            characterids.add(characterid)
            tropes.add(trope)
            imdbids.update(chatter.characterid_to_imdbids[characterid])
        characterids = sorted(characterids)
        tropes = sorted(tropes)
        imdbids = sorted(imdbids)

        # construct the character x trope labels array where value of positive entries (character portrays trope) is 1,
        # value of negative entries (character does not portray trope) is 0, and value of all other entries (we do not
        # know if character portrays or does not portray trope) is IGNORE_VALUE
        labels_arr = torch.full((len(characterids), len(tropes)), fill_value=100, dtype=torch.float32).cuda()
        for i, characterid in enumerate(characterids):
            for j, trope in enumerate(tropes):
                if (characterid, trope) in chatter.characterid_and_trope_to_label:
                    label = chatter.characterid_and_trope_to_label[(characterid, trope)]
                    labels_arr[i, j] = label

        # construct the character x trope labels mask array
        labels_mask = (labels_arr != 100)

        imdbid_to_index = {}
        for j, imdbid in enumerate(imdbids):
            imdb_characterids = imdbid_to_tensors[imdbid]["character-ids"]
            common_characterids = [characterid for characterid in characterids if characterid in imdb_characterids]
            component_index = [characterids.index(characterid) for characterid in common_characterids]
            imdb_index = [imdb_characterids.index(characterid) for characterid in common_characterids]
            imdbid_to_index[imdbid] = (j, component_index, imdb_index)

        trope_index = [alltropes.index(trope) for trope in tropes]

        componentid_to_tensors[componentid] = {"label": labels_arr,
                                               "mask": labels_mask,
                                               "characterids": characterids,
                                               "imdbids": imdbids,
                                               "tropes": tropes,
                                               "imdbid_to_index": imdbid_to_index,
                                               "trope_index": trope_index
                                               }
    return componentid_to_tensors

def model_component(model, classifier, componentid, chatter:Chatter, componentid_to_tensors, imdbid_to_tensors, 
                    trope_token_ids, loss_function):
    """run model and classifier on component"""
    # move the component data to gpu
    labels_arr = componentid_to_tensors[componentid]["label"]
    labels_mask = componentid_to_tensors[componentid]["mask"]
    characterids = componentid_to_tensors[componentid]["characterids"]
    imdbids = componentid_to_tensors[componentid]["imdbids"]
    tropes = componentid_to_tensors[componentid]["tropes"]
    component_trope_token_ids = trope_token_ids[componentid_to_tensors[componentid]["trope_index"]]

    # initialize the component character embeddings matrix
    # it is a zero matrix of shape characters x tropes x movies x hidden size
    character_embeddings = torch.zeros((len(characterids), len(tropes), len(imdbids), model.hidden_size), 
                                        dtype=torch.float32,
                                        requires_grad=componentid in chatter.train_componentids).cuda()

    # loop of component movies
    for imdbid in componentid_to_tensors[componentid]["imdbids"]:
        # move the movie data to gpu
        movie_token_ids = imdbid_to_tensors[imdbid]["token-ids"]
        movie_names_mask = imdbid_to_tensors[imdbid]["names-mask"]
        movie_mentions_mask = imdbid_to_tensors[imdbid]["mentions-mask"]
        movie_utterances_mask = imdbid_to_tensors[imdbid]["utterances-mask"]
        movie_mentions_character_ids = imdbid_to_tensors[imdbid]["mentions-character-ids"]
        movie_utterances_character_ids = imdbid_to_tensors[imdbid]["utterances-character-ids"]

        # run model
        # movie_character_embeddings = movie-characters x component-tropes x hidden-size
        movie_character_embeddings = model(movie_token_ids,
                                           movie_names_mask,
                                           movie_mentions_mask,
                                           movie_utterances_mask,
                                           movie_mentions_character_ids,
                                           movie_utterances_character_ids,
                                           component_trope_token_ids)

        # assign the character embeddings to the component character embeddings
        movie_index, characters_index, movie_characters_index = (
            componentid_to_tensors[componentid]["imdbid_to_index"][imdbid])
        character_embeddings[characters_index, :, movie_index] = (
            movie_character_embeddings[movie_characters_index].unsqueeze(dim=2))

    # classify whether character portrays trope
    logits = classifier(character_embeddings)
    logits = torch.max(logits, dim=2).values

    # compute loss
    labels = labels_arr[labels_mask]
    logits = logits[labels_mask]
    loss = loss_function(logits, labels)

    return loss, logits, labels

def train(model, classifier, optimizer, data_df, character_movie_map_df, tropes, trope_token_ids,
          tensors_dir, n_epochs, n_tropes_per_batch):
    print("initializing CHATTER data\n")
    chatter = Chatter(data_df, character_movie_map_df)

    print("creating batches")
    chatter.batch_components(n_tropes_per_batch)
    train_componentids = chatter.train_componentids
    dev_componentids = chatter.dev_componentids
    print("\n")

    print("reading movie tensors")
    imdbid_to_tensors = chatter.read_movie_data(tensors_dir)
    print("\n")

    print("creating batch tensors")
    componentid_to_tensors = get_component_tensors(chatter, imdbid_to_tensors, tropes, model.hidden_size)
    print("\n")

    # set up the loss function
    loss_function = BCEWithLogitsLoss()

    print("\n\nstart training\n\n")

    for epoch in range(n_epochs):
        # set the model to training mode
        model.train()
        classifier.train()

        # generate a random sequence in which components will be traversed
        componentids_index = np.random.permutation(len(train_componentids))

        # initialize the loss, logits, and labels arrays
        train_loss_arr = []
        valid_loss_arr = []
        valid_logits_arr = []
        valid_labels_arr = []

        tbar = tqdm.tqdm(componentids_index, unit="component", desc=f"training epoch {epoch + 1}")
        for i in tbar:
            # set the gradients to null
            optimizer.zero_grad()

            # get the componentid
            componentid = train_componentids[i]

            # run model on component
            loss, _, _ = model_component(model, classifier, componentid, chatter, componentid_to_tensors,
                                         imdbid_to_tensors, trope_token_ids, loss_function)

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

        tbar = tqdm.trange(dev_componentids, unit="component", desc=f"evaluating epoch {epoch + 1}")
        with torch.no_grad():
            for componentid in tbar:
                # run model on component
                loss, logits, labels = model_component(model, classifier, componentid, chatter, componentid_to_tensors,
                                                       imdbid_to_tensors, trope_token_ids, loss_function)

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