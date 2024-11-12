"""Train and evaluate the label dependent model on full story"""
import evaluation
from dataloaders.data import Chatter
from absl import logging

import collections
import math
import time
import tqdm
import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss

def run(encoder,
        model,
        classifier,
        partition,
        batch_imdbid,
        chatter:Chatter,
        batch_imdbid_to_tensors,
        tropes,
        trope_embeddings,
        trope_to_idx,
        loss_function,
        max_tropes_per_batch):
    """run model and classifier on batch movie"""
    if partition == "train":
        batch_tropes = chatter.trn_batch_imdbid_to_tropes[batch_imdbid]
    elif partition == "dev":
        batch_tropes = chatter.dev_batch_imdbid_to_tropes[batch_imdbid]
    elif partition == "test":
        batch_tropes = chatter.tst_batch_imdbid_to_tropes[batch_imdbid]

    n_subbatches = math.ceil(len(batch_tropes) / max_tropes_per_batch)
    logits_dict = collections.defaultdict(set)
    for i in range(n_subbatches):
        subbatch_tropes = tropes[i * max_tropes_per_batch: (i + 1) * max_tropes_per_batch]
        subbatch_tropes_idx = [trope_to_idx[trope] for trope in subbatch_tropes]
        subbatch_trope_embeddings = trope_embeddings[subbatch_tropes_idx]
        tensors = batch_imdbid_to_tensors[batch_imdbid]
        characterids = tensors["character-ids"]

        # embeddings = characters x tropes x hidden-size
        story_embeddings = encoder(tensors["token-ids"]).last_hidden_state
        embeddings = model(story_embeddings,
                           tensors["names-idx"],
                           tensors["mentions-idx"],
                           tensors["utterances-idx"],
                           tensors["mentions-character-ids"],
                           tensors["utterances-character-ids"],
                           subbatch_trope_embeddings)

        # logits = characters x tropes
        logits = classifier(embeddings)

        # create labels = characters x tropes
        subbatch_labels = torch.full((len(characterids), len(subbatch_tropes)), device="cuda:0", fill_value=100,
                                     dtype=torch.float32)
        for i, characterid in enumerate(characterids):
            for j, trope in enumerate(subbatch_tropes):
                if (characterid, trope) in chatter.characterid_and_trope_to_label:
                    subbatch_labels[i, j] = chatter.characterid_and_trope_to_label[(characterid, trope)]
                    logits_dict[(characterid, trope)].add(logits[i, j].item())

        # loss
        mask = subbatch_labels != 100
        logits = logits[mask]
        labels = subbatch_labels[mask]
        loss = loss_function(logits, labels)
        yield loss, logits_dict

def train(encoder,
          model,
          classifier,
          optimizer,
          data_df,
          character_movie_map_df,
          tropes,
          trope_embeddings,
          tensors_dir,
          n_epochs,
          max_tokens_per_batch,
          max_tropes_per_batch):
    logging.info("initializing CHATTER data")
    chatter = Chatter(data_df, character_movie_map_df)
    trope_to_idx = {trope:i for i, trope in enumerate(tropes)}
    imdbid_to_tensors = chatter.read_movie_data(tensors_dir)

    logging.info("\ncreating batches")
    batch_imdbid_to_tensors = chatter.batch_movies(imdbid_to_tensors, max_tokens_per_batch)

    # set up the loss function
    loss_function = BCEWithLogitsLoss()

    logging.info("\nstart training\n")
    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch + 1}")
        train_loss_arr = []
        valid_loss_arr = []
        valid_logits_dict = collections.defaultdict(set)

        # =================================================================
        # start training
        encoder.train()
        model.train()
        classifier.train()
        idx = np.random.permutation(len(chatter.trn_batch_imdbids))
        tbar = tqdm.tqdm(idx, unit="batch movie")
        for i in tbar:
            optimizer.zero_grad()
            batch_imdbid = chatter.trn_batch_imdbids[i]
            for loss, _ in run(encoder,
                               model,
                               classifier,
                               "train",
                               batch_imdbid,
                               chatter,
                               batch_imdbid_to_tensors,
                               tropes,
                               trope_embeddings,
                               trope_to_idx,
                               loss_function,
                               max_tropes_per_batch):
                loss.backward()
                optimizer.step()
                train_loss_arr.append(loss.item())
                tbar.set_description(f"train-loss = {loss.item():.4f}")
                optimizer.zero_grad()
        avg_train_loss = np.mean(train_loss_arr)
        logging.info(f"avg-train-loss = {avg_train_loss:.4f}")
        # =================================================================
        # end training

        # =================================================================
        # start evaluation
        encoder.eval()
        model.eval()
        classifier.eval()
        tbar = tqdm.tqdm(chatter.dev_batch_imdbids, unit="batch movie")
        with torch.no_grad():
            for batch_imdbid in tbar:
                for loss, logits_dict in run(encoder,
                                             model,
                                             classifier,
                                             "dev",
                                             batch_imdbid,
                                             chatter,
                                             batch_imdbid_to_tensors,
                                             tropes,
                                             trope_embeddings,
                                             trope_to_idx,
                                             loss_function,
                                             max_tropes_per_batch):
                    # save loss to loss array
                    valid_loss_arr.append(loss.item())
                    for key, vals in logits_dict.items():
                        valid_logits_dict[key].update(vals)
                    tbar.set_description(f"valid-loss = {loss.item():.4f}")
        valid_labels = []
        valid_logits = []
        for key, vals in valid_logits_dict.items():
            valid_labels.append(chatter.characterid_and_trope_to_label[key])
            valid_logits.append(max(vals))
        evaluation.evaluate(valid_logits, valid_labels)
        accuracy, precision, recall, f1 = evaluation.evaluate(valid_logits, valid_labels)
        avg_valid_loss = np.mean(valid_loss_arr)
        logging.info(f"avg-valid-loss = {avg_valid_loss:.4f} acc = {accuracy:.1f}, "
                     f"precision = {precision:.1f}, recall = {recall:.1f}, f1 = {f1:.1f}")
        # =================================================================
        # end evaluation

        logging.info(f"Epoch {epoch + 1} done\n")