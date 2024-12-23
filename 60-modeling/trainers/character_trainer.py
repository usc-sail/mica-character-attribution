"""Train and evaluate the label dependent model on full story"""
import utils
import evaluation
from dataloaders.data import Chatter
from absl import logging

import math
import tqdm
import torch
import random
import numpy as np
from torch.nn import BCEWithLogitsLoss

def run(lbl,
        encoder,
        model,
        classifier,
        tensors,
        chatter:Chatter,
        trope_embeddings,
        trope_to_idx,
        loss_function,
        max_tropes_per_batch):
    """run model and classifier on character segments batch"""
    batch_tropes = tensors["tropes"]
    n_subbatches = math.ceil(len(batch_tropes)/max_tropes_per_batch)
    logits_dict = {}
    for i in range(n_subbatches):
        subbatch_tropes = batch_tropes[i * max_tropes_per_batch: (i + 1) * max_tropes_per_batch]
        subbatch_tropes_idx = [trope_to_idx[trope] for trope in subbatch_tropes]
        subbatch_trope_embeddings = trope_embeddings[subbatch_tropes_idx] # [t, d]
        characterids = tensors["character-ids"]

        # run model
        segment_mask = torch.any(tensors["batch-token-ids"] != 0, dim=2) # [k, b]
        segment_mask = torch.log(segment_mask) # [k, b]
        batch_token_ids = tensors["batch-token-ids"] # [k, b, l]
        batch_mask = tensors["batch-mask"] # [k, b, l]
        n_characters, n_blocks, seqlen = batch_token_ids.shape
        batch_token_ids = batch_token_ids.reshape(-1, seqlen) # [kb, l]
        batch_mask = batch_mask.reshape(-1, seqlen) # [kb, l]
        segment_embeddings = encoder(batch_token_ids, attention_mask=batch_mask).last_hidden_state # [kb, l, d]
        segment_embeddings = segment_embeddings.reshape(n_characters, n_blocks, seqlen, -1) # [k, b, l, d]
        if lbl:
            embeddings = model(segment_embeddings,
                               segment_mask,
                               tensors["batch-names-idx"],
                               tensors["batch-mentions-idx"],
                               tensors["batch-utterances-idx"],
                               subbatch_trope_embeddings) # [k, t, d]
        else:
            character_embeddings = model(segment_embeddings,
                                         segment_mask,
                                         tensors["batch-names-idx"],
                                         tensors["batch-mentions-idx"],
                                         tensors["batch-utterances-idx"]) # [k, d]
            embeddings = utils.cartesian_concatenate(character_embeddings, subbatch_trope_embeddings) # [k, t, 2d]
        logits = classifier(embeddings) # [k, t]

        # create labels
        subbatch_labels = torch.full((len(characterids), len(subbatch_tropes)), device="cuda:0", fill_value=100,
                                     dtype=torch.float32) # [k, t]
        for i, characterid in enumerate(characterids):
            for j, trope in enumerate(subbatch_tropes):
                if (characterid, trope) in chatter.characterid_and_trope_to_label:
                    subbatch_labels[i, j] = chatter.characterid_and_trope_to_label[(characterid, trope)]
                    logits_dict[(characterid, trope)] = logits[i, j].item()

        # calculate loss
        mask = subbatch_labels != 100
        logits = logits[mask]
        labels = subbatch_labels[mask]
        loss = loss_function(logits, labels)
        yield loss, logits_dict

def train(lbl,
          encoder,
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
          max_tropes_per_batch,
          max_grad_norm,
          batches_per_epoch,
          batches_eval):
    logging.info("initializing CHATTER data")
    chatter = Chatter(data_df, character_movie_map_df)
    trope_to_idx = {trope:i for i, trope in enumerate(tropes)}
    characterid_to_tensors = chatter.read_character_data(tensors_dir)

    logging.info("\ncreating batches")
    trn_batches, dev_batches, tst_batches = chatter.batch_characters(characterid_to_tensors, max_tokens_per_batch)

    # set up the loss function
    loss_function = BCEWithLogitsLoss()
    tot_trn_batches = batches_per_epoch * n_epochs
    n_trn_batches = len(trn_batches)
    n_iter = tot_trn_batches // n_trn_batches
    rem_n_batches = tot_trn_batches % n_trn_batches
    idx = ([i for _ in range(n_iter) for i in np.random.permutation(n_trn_batches)]
           + np.random.choice(n_trn_batches, rem_n_batches, replace=False).tolist())
    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch + 1}")
        train_loss_arr = []
        valid_loss_arr = []
        valid_logits_dict = {}

        # =================================================================
        # start training
        encoder.train()
        model.train()
        classifier.train()
        epoch_idx = idx[epoch * batches_per_epoch: (epoch + 1) * batches_per_epoch]
        tbar = tqdm.tqdm(epoch_idx, unit="batch segment")
        nsamples = 0
        for i in tbar:
            optimizer.zero_grad()
            tensors = trn_batches[i]
            for loss, logits_dict in run(lbl,
                                         encoder,
                                         model,
                                         classifier,
                                         tensors,
                                         chatter,
                                         trope_embeddings,
                                         trope_to_idx,
                                         loss_function,
                                         max_tropes_per_batch):
                loss.backward()
                optimizer.step()
                nsamples += len(logits_dict)
                train_loss_arr.append(loss.item())
                running_avg_train_loss = np.mean(train_loss_arr[-10:])
                avg_train_loss = np.mean(train_loss_arr)
                tbar.set_description(f"train-loss: batch={loss.item():.4f} running-avg={running_avg_train_loss:.4f} "
                                     f"avg={avg_train_loss:.4f}")
                optimizer.zero_grad()
        avg_train_loss = np.mean(train_loss_arr)
        logging.info(f"avg-train-loss = {avg_train_loss:.4f}")
        logging.info(f"nsamples trained = {nsamples}")
        # =================================================================
        # end training

        # =================================================================
        # start evaluation
        encoder.eval()
        model.eval()
        classifier.eval()
        eval_batches = random.sample(dev_batches, min(batches_eval, len(dev_batches)))
        tbar = tqdm.tqdm(eval_batches, unit="batch segment")
        with torch.no_grad():
            for tensors in tbar:
                for loss, logits_dict in run(lbl,
                                             encoder,
                                             model,
                                             classifier,
                                             tensors,
                                             chatter,
                                             trope_embeddings,
                                             trope_to_idx,
                                             loss_function,
                                             max_tropes_per_batch):
                    valid_loss_arr.append(loss.item())
                    avg_valid_loss = np.mean(valid_loss_arr)
                    valid_logits_dict.update(logits_dict)
                    tbar.set_description(f"valid-loss: batch={loss.item():.4f} avg={avg_valid_loss:.4f}")
        valid_labels = []
        valid_logits = []
        for key, val in valid_logits_dict.items():
            valid_labels.append(chatter.characterid_and_trope_to_label[key])
            valid_logits.append(val)
        valid_logits = np.array(valid_logits)
        valid_labels = np.array(valid_labels)
        accuracy, precision, recall, f1 = evaluation.evaluate(valid_logits, valid_labels)
        avg_valid_loss = np.mean(valid_loss_arr)
        logging.info(f"avg-valid-loss = {avg_valid_loss:.4f} accuracy = {accuracy:.1f}, "
                     f"precision = {precision:.1f}, recall = {recall:.1f}, f1 = {f1:.1f}")
        # =================================================================
        # end evaluation

        logging.info(f"Epoch {epoch + 1} done\n")