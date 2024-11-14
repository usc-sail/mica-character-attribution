"""Functions for data preprocessing"""
import collections
import math
import numpy as np
import pandas as pd
import os
import random
import torch
import tqdm
from absl import logging

class Chatter:

    def __init__(self, data_df, character_movie_map_df) -> None:
        data_df = data_df.merge(character_movie_map_df[["character", "component"]].drop_duplicates(),
                                how="left", on="character")
        imdbid_and_charactername_to_characterid = {}
        characterid_to_imdbids = collections.defaultdict(set)
        imdbid_to_characterids = collections.defaultdict(set)
        characterid_and_trope_to_label = {}

        for characterid, imdbid, name in (character_movie_map_df[["character", "imdb-id", "name"]]
                                          .itertuples(index=False, name=None)):
            imdbid_and_charactername_to_characterid[(imdbid, name)] = characterid
            characterid_to_imdbids[characterid].add(imdbid)
            imdbid_to_characterids[imdbid].add(characterid)

        for characterid, trope, tvtrope_label, label in (data_df[["character", "trope", "tvtrope-label", "label"]]
                                                         .itertuples(index=False, name=None)):
            if pd.notna(label):
                characterid_and_trope_to_label[(characterid, trope)] = label
            else:
                characterid_and_trope_to_label[(characterid, trope)] = tvtrope_label

        self.data_df = data_df
        self.map_df = character_movie_map_df
        self.imdbid_and_charactername_to_characterid = imdbid_and_charactername_to_characterid
        self.characterid_to_imdbids = self.sort_dictionary(characterid_to_imdbids)
        self.imdbid_to_characterids = self.sort_dictionary(imdbid_to_characterids)
        self.characterid_and_trope_to_label = characterid_and_trope_to_label

    def sort_dictionary(self, dictionary):
        return {key: sorted(values) for key, values in dictionary.items()}

    def read_movie_data(self, tensors_dir):
        imdbids = self.map_df["imdb-id"].unique()
        imdbid_to_tensors = {}
        for imdbid in tqdm.tqdm(imdbids, desc="reading movie tensors", unit="movie"):
            movie_dir = os.path.join(tensors_dir, imdbid)
            characters_file = os.path.join(movie_dir, "characters.txt")
            token_ids_file = os.path.join(movie_dir, "token-ids.pt")
            names_idx_file = os.path.join(movie_dir, "names-idx.pt")
            mentions_idx_file = os.path.join(movie_dir, "mentions-idx.pt")
            utterances_idx_file = os.path.join(movie_dir, "utterances-idx.pt")
            mention_character_ids_file = os.path.join(movie_dir, "mention-character-ids.pt")
            utterance_character_ids_file = os.path.join(movie_dir, "utterance-character-ids.pt")
            characternames = open(characters_file).read().split("\n")
            characterids = [self.imdbid_and_charactername_to_characterid[(imdbid, name)] for name in characternames]
            token_ids = torch.load(token_ids_file, weights_only=True).cuda()
            names_idx = torch.load(names_idx_file, weights_only=True).cuda()
            mentions_idx = torch.load(mentions_idx_file, weights_only=True).cuda()
            utterances_idx = torch.load(utterances_idx_file, weights_only=True).cuda()
            mention_character_ids = torch.load(mention_character_ids_file, weights_only=True).cuda()
            utterance_character_ids = torch.load(utterance_character_ids_file, weights_only=True).cuda()
            imdbid_to_tensors[imdbid] = {"character-ids": characterids,
                                         "token-ids": token_ids,
                                         "names-idx": names_idx,
                                         "mentions-idx": mentions_idx,
                                         "utterances-idx": utterances_idx,
                                         "mentions-character-ids": mention_character_ids,
                                         "utterances-character-ids": utterance_character_ids,
                                         }
        return imdbid_to_tensors

    def tensorshape(self, tensor):
        return "[" + ",".join(f"{d:5d}" for d in tensor.shape) + "]"

    def batch_movies(self, imdbid_to_tensors, max_n_tokens):
        n_seq_tokens = next(iter(imdbid_to_tensors.values()))["token-ids"].shape[1]
        max_n_seqs = math.ceil(max_n_tokens / n_seq_tokens)
        logging.info(f"number of sequence tokens = {n_seq_tokens}")
        logging.info(f"maximum number of sequences per batch = {max_n_seqs}")

        characterid_to_batch_imdbids = collections.defaultdict(set)
        batch_imdbid_to_characterids = collections.defaultdict(set)
        batch_imdbid_to_tensors = {}

        n_missing_batch_mentions = 0
        n_missing_batch_utterances = 0

        for imdbid, tensors_dict in tqdm.tqdm(imdbid_to_tensors.items(), desc="batching movies", unit="movie"):
            n_seqs = tensors_dict["token-ids"].shape[0]
            n_batches = math.ceil(n_seqs / max_n_seqs)

            character_ids = tensors_dict["character-ids"]
            token_ids = tensors_dict["token-ids"]
            names_idx = tensors_dict["names-idx"]
            mentions_idx = tensors_dict["mentions-idx"]
            utterances_idx = tensors_dict["utterances-idx"]
            mention_character_ids = tensors_dict["mentions-character-ids"]
            utterance_character_ids = tensors_dict["utterances-character-ids"]

            n_batch_mentions = 0
            n_batch_utterances = 0

            for i in range(n_batches):
                batch_imdbid = f"{imdbid}-{i}"
                batch_character_ids = character_ids
                batch_token_ids = token_ids[i * max_n_seqs : (i + 1) * max_n_seqs]
                batch_names_idx = names_idx

                if mentions_idx.nelement() > 0:
                    batch_mentions_mask = ((mentions_idx[:,0] >= i * max_n_seqs * n_seq_tokens)
                                           & (mentions_idx[:,1] <= (i + 1) * max_n_seqs * n_seq_tokens))
                    batch_mentions_idx = mentions_idx[batch_mentions_mask] - (i * max_n_seqs * n_seq_tokens)
                    batch_mention_character_ids = mention_character_ids[batch_mentions_mask]
                else:
                    batch_mentions_idx = mentions_idx
                    batch_mention_character_ids = mention_character_ids
                n_batch_mentions += len(batch_mentions_idx)

                if utterances_idx.nelement() > 0:
                    batch_utterances_mask = ((utterances_idx[:,0] >= i * max_n_seqs * n_seq_tokens)
                                             & (utterances_idx[:,1] <= (i + 1) * max_n_seqs * n_seq_tokens))
                    batch_utterances_idx = utterances_idx[batch_utterances_mask] - (i * max_n_seqs * n_seq_tokens)
                    batch_utterance_character_ids = utterance_character_ids[batch_utterances_mask]
                else:
                    batch_utterances_idx = utterances_idx
                    batch_utterance_character_ids = utterance_character_ids
                n_batch_utterances += len(batch_utterances_idx)

                for characterid in self.imdbid_to_characterids[imdbid]:
                    characterid_to_batch_imdbids[characterid].add(batch_imdbid)
                    batch_imdbid_to_characterids[batch_imdbid].add(characterid)

                batch_imdbid_to_tensors[batch_imdbid] = {"character-ids": batch_character_ids,
                                                         "token-ids": batch_token_ids,
                                                         "names-idx": batch_names_idx,
                                                         "mentions-idx": batch_mentions_idx,
                                                         "utterances-idx": batch_utterances_idx,
                                                         "mentions-character-ids": batch_mention_character_ids,
                                                         "utterances-character-ids": batch_utterance_character_ids}
            n_missing_batch_mentions += len(mentions_idx) - n_batch_mentions
            n_missing_batch_utterances += len(utterances_idx) - n_batch_utterances

        logging.info(f"{n_missing_batch_mentions} mentions missed in batches")
        logging.info(f"{n_missing_batch_utterances} utterances missed in batches")

        self.batch_imdbid_to_characterids = self.sort_dictionary(batch_imdbid_to_characterids)
        self.characterid_to_batch_imdbids = self.sort_dictionary(characterid_to_batch_imdbids)

        trn_batch_imdbid_to_tropes = collections.defaultdict(set)
        dev_batch_imdbid_to_tropes = collections.defaultdict(set)
        tst_batch_imdbid_to_tropes = collections.defaultdict(set)
        trn_batch_imdbids = set()
        dev_batch_imdbids = set()
        tst_batch_imdbids = set()

        for characterid, trope, partition in (self.data_df[["character", "trope", "partition"]]
                                              .itertuples(index=False, name=None)):
            if partition == "train":
                batch_imdbids = self.characterid_to_batch_imdbids[characterid]
                for batch_imdbid in batch_imdbids:
                    trn_batch_imdbid_to_tropes[batch_imdbid].add(trope)
                trn_batch_imdbids.update(batch_imdbids)
            elif partition == "dev":
                batch_imdbids = self.characterid_to_batch_imdbids[characterid]
                for batch_imdbid in batch_imdbids:
                    dev_batch_imdbid_to_tropes[batch_imdbid].add(trope)
                dev_batch_imdbids.update(batch_imdbids)
            elif partition == "test":
                batch_imdbids = self.characterid_to_batch_imdbids[characterid]
                for batch_imdbid in batch_imdbids:
                    tst_batch_imdbid_to_tropes[batch_imdbid].add(trope)
                tst_batch_imdbids.update(batch_imdbids)

        self.trn_batch_imdbid_to_tropes = self.sort_dictionary(trn_batch_imdbid_to_tropes)
        self.dev_batch_imdbid_to_tropes = self.sort_dictionary(dev_batch_imdbid_to_tropes)
        self.tst_batch_imdbid_to_tropes = self.sort_dictionary(tst_batch_imdbid_to_tropes)
        self.trn_batch_imdbids = sorted(trn_batch_imdbids)
        self.dev_batch_imdbids = sorted(dev_batch_imdbids)
        self.tst_batch_imdbids = sorted(tst_batch_imdbids)
        logging.info(f"{len(self.trn_batch_imdbids)} train batches")

        return batch_imdbid_to_tensors