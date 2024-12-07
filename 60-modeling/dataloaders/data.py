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
    
    def read_character_data(self, tensors_dir):
        characterid_to_tensors = {}
        for characterid in os.listdir(tensors_dir):
            token_ids_file = os.path.join(tensors_dir, characterid, "token-ids.pt")
            if os.path.exists(token_ids_file):
                names_idx_file = os.path.join(tensors_dir, characterid, "names-idx.pt")
                mentions_idx_file = os.path.join(tensors_dir, characterid, "mentions-idx.pt")
                utterances_idx_file = os.path.join(tensors_dir, characterid, "utterances-idx.pt")
                token_ids = torch.load(token_ids_file, weights_only=True).cuda()
                names_idx = torch.load(names_idx_file, weights_only=True).cuda()
                mentions_idx = torch.load(mentions_idx_file, weights_only=True).cuda()
                utterances_idx = torch.load(utterances_idx_file, weights_only=True).cuda()
                characterid_to_tensors[characterid] = {"token-ids": token_ids,
                                                       "names-idx": names_idx,
                                                       "mentions-idx": mentions_idx,
                                                       "utterances-idx": utterances_idx}
        return characterid_to_tensors

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
        trn_batch_imdbid_to_nsamples = collections.defaultdict(int)
        trn_characterid_to_batch_imdbids = collections.defaultdict(set)
        trn_characterid_and_trope_to_nbatch_imdbids = collections.defaultdict(int)
        dev_batch_imdbid_to_nsamples = collections.defaultdict(int)
        dev_characterid_to_batch_imdbids = collections.defaultdict(set)
        dev_characterid_and_trope_to_nbatch_imdbids = collections.defaultdict(int)
        tst_batch_imdbid_to_nsamples = collections.defaultdict(int)
        tst_characterid_to_batch_imdbids = collections.defaultdict(set)
        tst_characterid_and_trope_to_nbatch_imdbids = collections.defaultdict(int)

        for characterid, trope, partition in (self.data_df[["character", "trope", "partition"]]
                                              .itertuples(index=False, name=None)):
            if partition == "train":
                batch_imdbids = self.characterid_to_batch_imdbids[characterid]
                for batch_imdbid in batch_imdbids:
                    trn_batch_imdbid_to_tropes[batch_imdbid].add(trope)
                    trn_batch_imdbid_to_nsamples[batch_imdbid] += 1
                    trn_characterid_to_batch_imdbids[characterid].add(batch_imdbid)
                    trn_characterid_and_trope_to_nbatch_imdbids[(characterid, trope)] += 1
                trn_batch_imdbids.update(batch_imdbids)
            elif partition == "dev":
                batch_imdbids = self.characterid_to_batch_imdbids[characterid]
                for batch_imdbid in batch_imdbids:
                    dev_batch_imdbid_to_tropes[batch_imdbid].add(trope)
                    dev_batch_imdbid_to_nsamples[batch_imdbid] += 1
                    dev_characterid_to_batch_imdbids[characterid].add(batch_imdbid)
                    dev_characterid_and_trope_to_nbatch_imdbids[(characterid, trope)] += 1
                dev_batch_imdbids.update(batch_imdbids)
            elif partition == "test":
                batch_imdbids = self.characterid_to_batch_imdbids[characterid]
                for batch_imdbid in batch_imdbids:
                    tst_batch_imdbid_to_tropes[batch_imdbid].add(trope)
                    tst_batch_imdbid_to_nsamples[batch_imdbid] += 1
                    tst_characterid_to_batch_imdbids[characterid].add(batch_imdbid)
                    tst_characterid_and_trope_to_nbatch_imdbids[(characterid, trope)] += 1
                tst_batch_imdbids.update(batch_imdbids)

        self.trn_batch_imdbid_to_tropes = self.sort_dictionary(trn_batch_imdbid_to_tropes)
        self.dev_batch_imdbid_to_tropes = self.sort_dictionary(dev_batch_imdbid_to_tropes)
        self.tst_batch_imdbid_to_tropes = self.sort_dictionary(tst_batch_imdbid_to_tropes)
        self.trn_batch_imdbids = sorted(trn_batch_imdbids)
        self.dev_batch_imdbids = sorted(dev_batch_imdbids)
        self.tst_batch_imdbids = sorted(tst_batch_imdbids)

        n_batches = len(trn_batch_imdbids)
        avg_nsamples_per_batch_imdbid = np.mean(list(trn_batch_imdbid_to_nsamples.values()))
        n_characterids = len(trn_characterid_to_batch_imdbids)
        n_characterids_one_batch_imdbid = sum([len(batch_imdbids) == 1
                                               for batch_imdbids in trn_characterid_to_batch_imdbids.values()])
        p_one = 100 * n_characterids_one_batch_imdbid/n_characterids
        n_characterids_two_batch_imdbids = sum([len(batch_imdbids) == 2
                                                for batch_imdbids in trn_characterid_to_batch_imdbids.values()])
        p_two = 100 * n_characterids_two_batch_imdbids/n_characterids
        n_characterids_more_batch_imdbids = sum([len(batch_imdbids) > 2
                                                 for batch_imdbids in trn_characterid_to_batch_imdbids.values()])
        p_more = 100 * n_characterids_more_batch_imdbids/n_characterids
        n_samples = len(trn_characterid_and_trope_to_nbatch_imdbids)
        n_samples_one_batch_imdbid = sum([nbatch_imdbids == 1
                                          for nbatch_imdbids in trn_characterid_and_trope_to_nbatch_imdbids.values()])
        q_one = 100 * n_samples_one_batch_imdbid / n_samples
        n_samples_two_batch_imdbids = sum([nbatch_imdbids == 2
                                           for nbatch_imdbids in trn_characterid_and_trope_to_nbatch_imdbids.values()])
        q_two = 100 * n_samples_two_batch_imdbids / n_samples
        n_samples_more_batch_imdbids = sum([nbatch_imdbids > 2
                                            for nbatch_imdbids in trn_characterid_and_trope_to_nbatch_imdbids.values()])
        q_more = 100 * n_samples_more_batch_imdbids / n_samples
        logging.info("TRAIN:")
        logging.info(f"\t{n_batches} batch imdbids, "
                     f"avg {avg_nsamples_per_batch_imdbid:.1f} samples per batch imdbid")
        logging.info(f"\t{n_characterids} characters, {n_characterids_one_batch_imdbid} ({p_one:.1f}%), "
                     f"{n_characterids_two_batch_imdbids} ({p_two:.1f}%) and "
                     f"{n_characterids_more_batch_imdbids} ({p_more:.1f}%) appear in 1, 2, >2 batch imdbids")
        logging.info(f"\t{n_samples} samples, {n_samples_one_batch_imdbid} ({q_one:.1f}%), "
                     f"{n_samples_two_batch_imdbids} ({q_two:.1f}%) and "
                     f"{n_samples_more_batch_imdbids} ({q_more:.1f}%) appear in 1, 2, >2 batch imdbids")

        n_batches = len(dev_batch_imdbids)
        avg_nsamples_per_batch_imdbid = np.mean(list(dev_batch_imdbid_to_nsamples.values()))
        n_characterids = len(dev_characterid_to_batch_imdbids)
        n_characterids_one_batch_imdbid = sum([len(batch_imdbids) == 1
                                               for batch_imdbids in dev_characterid_to_batch_imdbids.values()])
        p_one = 100 * n_characterids_one_batch_imdbid/n_characterids
        n_characterids_two_batch_imdbids = sum([len(batch_imdbids) == 2
                                                for batch_imdbids in dev_characterid_to_batch_imdbids.values()])
        p_two = 100 * n_characterids_two_batch_imdbids/n_characterids
        n_characterids_more_batch_imdbids = sum([len(batch_imdbids) > 2
                                                 for batch_imdbids in dev_characterid_to_batch_imdbids.values()])
        p_more = 100 * n_characterids_more_batch_imdbids/n_characterids
        n_samples = len(dev_characterid_and_trope_to_nbatch_imdbids)
        n_samples_one_batch_imdbid = sum([nbatch_imdbids == 1
                                          for nbatch_imdbids in dev_characterid_and_trope_to_nbatch_imdbids.values()])
        q_one = 100 * n_samples_one_batch_imdbid / n_samples
        n_samples_two_batch_imdbids = sum([nbatch_imdbids == 2
                                           for nbatch_imdbids in dev_characterid_and_trope_to_nbatch_imdbids.values()])
        q_two = 100 * n_samples_two_batch_imdbids / n_samples
        n_samples_more_batch_imdbids = sum([nbatch_imdbids > 2
                                            for nbatch_imdbids in dev_characterid_and_trope_to_nbatch_imdbids.values()])
        q_more = 100 * n_samples_more_batch_imdbids / n_samples
        logging.info("DEV:")
        logging.info(f"\t{n_batches} batch imdbids, "
                     f"avg {avg_nsamples_per_batch_imdbid:.1f} samples per batch imdbid")
        logging.info(f"\t{n_characterids} characters, {n_characterids_one_batch_imdbid} ({p_one:.1f}%), "
                     f"{n_characterids_two_batch_imdbids} ({p_two:.1f}%) and "
                     f"{n_characterids_more_batch_imdbids} ({p_more:.1f}%) appear in 1, 2, >2 batch imdbids")
        logging.info(f"\t{n_samples} samples, {n_samples_one_batch_imdbid} ({q_one:.1f}%), "
                     f"{n_samples_two_batch_imdbids} ({q_two:.1f}%) and "
                     f"{n_samples_more_batch_imdbids} ({q_more:.1f}%) appear in 1, 2, >2 batch imdbids")

        n_batches = len(tst_batch_imdbids)
        avg_nsamples_per_batch_imdbid = np.mean(list(tst_batch_imdbid_to_nsamples.values()))
        n_characterids = len(tst_characterid_to_batch_imdbids)
        n_characterids_one_batch_imdbid = sum([len(batch_imdbids) == 1
                                               for batch_imdbids in tst_characterid_to_batch_imdbids.values()])
        p_one = 100 * n_characterids_one_batch_imdbid/n_characterids
        n_characterids_two_batch_imdbids = sum([len(batch_imdbids) == 2
                                                for batch_imdbids in tst_characterid_to_batch_imdbids.values()])
        p_two = 100 * n_characterids_two_batch_imdbids/n_characterids
        n_characterids_more_batch_imdbids = sum([len(batch_imdbids) > 2
                                                 for batch_imdbids in tst_characterid_to_batch_imdbids.values()])
        p_more = 100 * n_characterids_more_batch_imdbids/n_characterids
        n_samples = len(tst_characterid_and_trope_to_nbatch_imdbids)
        n_samples_one_batch_imdbid = sum([nbatch_imdbids == 1
                                          for nbatch_imdbids in tst_characterid_and_trope_to_nbatch_imdbids.values()])
        q_one = 100 * n_samples_one_batch_imdbid / n_samples
        n_samples_two_batch_imdbids = sum([nbatch_imdbids == 2
                                           for nbatch_imdbids in tst_characterid_and_trope_to_nbatch_imdbids.values()])
        q_two = 100 * n_samples_two_batch_imdbids / n_samples
        n_samples_more_batch_imdbids = sum([nbatch_imdbids > 2
                                            for nbatch_imdbids in tst_characterid_and_trope_to_nbatch_imdbids.values()])
        q_more = 100 * n_samples_more_batch_imdbids / n_samples
        logging.info("TEST:")
        logging.info(f"\t{n_batches} batch imdbids, "
                     f"avg {avg_nsamples_per_batch_imdbid:.1f} samples per batch imdbid")
        logging.info(f"\t{n_characterids} characters, {n_characterids_one_batch_imdbid} ({p_one:.1f}%), "
                     f"{n_characterids_two_batch_imdbids} ({p_two:.1f}%) and "
                     f"{n_characterids_more_batch_imdbids} ({p_more:.1f}%) appear in 1, 2, >2 batch imdbids")
        logging.info(f"\t{n_samples} samples, {n_samples_one_batch_imdbid} ({q_one:.1f}%), "
                     f"{n_samples_two_batch_imdbids} ({q_two:.1f}%) and "
                     f"{n_samples_more_batch_imdbids} ({q_more:.1f}%) appear in 1, 2, >2 batch imdbids")

        return batch_imdbid_to_tensors

    def batch_characters(self, characterid_to_tensors, max_n_tokens):
        # find the sequence length of the token-ids tensors, data type of the token-ids and the data type of the 
        # mentions/utterances-idx tensors, and the device which holds the tensors
        for tensors_dict in characterid_to_tensors.values():
            token_ids = tensors_dict["token-ids"]
            mentions_idx = tensors_dict["mentions-idx"]
            seqlen = token_ids.shape[1]
            dtype_ids = token_ids.dtype
            dtype_idx = mentions_idx.dtype
            device = token_ids.device
            break

        # calculate the maximum number of sequences (1 sequence = 1 block) allowed in a batch based on the maximum 
        # number of tokens allowed in a batch
        max_n_blocks_batch = math.ceil(max_n_tokens / seqlen)

        # find the set of tropes portrayed or not portrayed by each character for each partition
        # train characterids are mutually exclusive from dev and test characterids
        # test characterids is a subset of dev characterids 
        # but the test characterids are tested on a different set of tropes
        trn_characterid_to_tropes = collections.defaultdict(set)
        dev_characterid_to_tropes = collections.defaultdict(set)
        tst_characterid_to_tropes = collections.defaultdict(set)
        for characterid, trope, partition in (self.data_df[["character", "trope", "partition"]]
                                              .itertuples(index=False, name=None)):
            if characterid in characterid_to_tensors:
                if partition == "train":
                    trn_characterid_to_tropes[characterid].add(trope)
                elif partition == "dev":
                    dev_characterid_to_tropes[characterid].add(trope)
                elif partition == "test":
                    tst_characterid_to_tropes[characterid].add(trope)

        character_batches_arr = []
        characterid_to_tropes_arr = [trn_characterid_to_tropes, dev_characterid_to_tropes, tst_characterid_to_tropes]
        partitions = ["TRAIN", "DEV", "TEST"]
        for characterid_to_tropes, partition in zip(characterid_to_tropes_arr, partitions):
            characterids = sorted(characterid_to_tropes.keys())
            random.seed(0)
            random.shuffle(characterids)

            # calculate the number of sequences in every character segment
            n_blocks = []
            for characterid in characterids:
                n_blocks.append(characterid_to_tensors[characterid]["token-ids"].shape[0])
            sort_index = np.argsort(n_blocks, stable=True)

            i = 0
            character_batches = []
            n_samples = 0
            n_characters_with_n_blocks_grt_max_n_blocks = 0
            n_tropes_batch_arr = []
            frac_pos_tropes_batch_arr = []

            while i < len(characterids):
                max_n_blocks_character = n_blocks[sort_index[i]]

                # if the character's segments contain more than the allowed number of sequences of a batch, then 
                # truncate the character's segments to the allowed number of sequences
                # the batch will contain exactly one character
                if max_n_blocks_character > max_n_blocks_batch:
                    n_characters_with_n_blocks_grt_max_n_blocks += 1
                    characterid = characterids[sort_index[i]]
                    tensors_dict = characterid_to_tensors[characterid]

                    # token_ids = [1, b, l]
                    # names_idx = [1, 2]
                    # mentions_idx = [1, m, 2]
                    # utterances_idx = [1, u, 2]
                    token_ids = tensors_dict["token-ids"][:max_n_blocks_batch].unsqueeze(1)
                    names_idx = tensors_dict["names-idx"]
                    mentions_idx = tensors_dict["mentions-idx"]
                    utterances_idx = tensors_dict["utterances-idx"]
                    mentions_idx = mentions_idx[mentions_idx[:, 1] <= max_n_blocks_batch * seqlen].unsqueeze(1)
                    utterances_idx = utterances_idx[utterances_idx[:, 1] <= max_n_blocks_batch * seqlen].unsqueeze(1)
                    tropes = sorted(characterid_to_tropes[characterid])
                    n_tropes_batch_arr.append(len(tropes))
                    n_pos_tropes_batch = sum([self.characterid_and_trope_to_label[(characterid, trope)] == 1
                                              for trope in tropes])
                    frac_pos_tropes_batch_arr.append(n_pos_tropes_batch/len(tropes))
                    n_samples += len(tropes)
                    character_batches.append({"character-ids": [characterid],
                                              "batch-token-ids": token_ids,
                                              "batch-names-idx": names_idx,
                                              "batch-mentions-idx": mentions_idx,
                                              "batch-utterances-idx": utterances_idx,
                                              "tropes": tropes})
                    i += 1

                else:
                    # calculate the maximum number of blocks allowed per character in a batch
                    j = i + 1
                    while j < len(characterids) and max_n_blocks_character * (j - i) <= max_n_blocks_batch:
                        max_n_blocks_character = max(n_blocks[sort_index[j]], max_n_blocks_character)
                        j += 1
                    batch_characterids = [characterids[sort_index[k]] for k in range(i, j)]
                    max_n_blocks_character = -1
                    max_n_mentions_character = -1
                    max_n_utterances_character = -1
                    token_ids_arr = []
                    names_idx_arr = []
                    mentions_idx_arr = []
                    utterances_idx_arr = []

                    for characterid in batch_characterids:
                        tensors_dict = characterid_to_tensors[characterid]
                        token_ids = tensors_dict["token-ids"]
                        names_idx = tensors_dict["names-idx"]
                        mentions_idx = tensors_dict["mentions-idx"]
                        utterances_idx = tensors_dict["utterances-idx"]
                        max_n_blocks_character = max(token_ids.shape[0], max_n_blocks_character)
                        max_n_mentions_character = max(mentions_idx.shape[0], max_n_mentions_character)
                        max_n_utterances_character = max(utterances_idx.shape[0], max_n_utterances_character)
                        token_ids_arr.append(token_ids)
                        names_idx_arr.append(names_idx)
                        mentions_idx_arr.append(mentions_idx)
                        utterances_idx_arr.append(utterances_idx)

                    # pad the token-ids, mentions-idx and utterances-idx to get [n_characters, n_blocks, ...] shape
                    n = j - i
                    for k in range(n):
                        n_padding_blocks = max_n_blocks_character - token_ids_arr[k].shape[0]
                        n_padding_mentions_idx = max_n_mentions_character - mentions_idx_arr[k].shape[0]
                        n_padding_utterances_idx = max_n_utterances_character - utterances_idx_arr[k].shape[0]
                        if n_padding_blocks > 0:
                            padding_blocks = torch.zeros((n_padding_blocks, seqlen), dtype=dtype_ids, device=device)
                            token_ids_arr[k] = torch.cat([token_ids_arr[k], padding_blocks], dim=0)
                        if n_padding_mentions_idx > 0:
                            padding_mentions_idx = torch.zeros((n_padding_mentions_idx, 2), dtype=dtype_idx,
                                                               device=device)
                            mentions_idx_arr[k] = torch.cat([mentions_idx_arr[k], padding_mentions_idx], dim=0)
                        if n_padding_utterances_idx > 0:
                            padding_utterances_idx = torch.zeros((n_padding_utterances_idx, 2), dtype=dtype_idx,
                                                                 device=device)
                            utterances_idx_arr[k] = torch.cat([utterances_idx_arr[k], padding_utterances_idx], dim=0)
                    batch_token_ids = torch.cat(token_ids_arr, dim=0).reshape(n, max_n_blocks_character, seqlen)
                    batch_names_idx = torch.cat(names_idx_arr, dim=0).reshape(n, 2)
                    batch_mentions_idx = torch.cat(mentions_idx_arr, dim=0).reshape(n, max_n_mentions_character, 2)
                    batch_utterances_idx = torch.cat(utterances_idx_arr, dim=0).reshape(
                        n, max_n_utterances_character, 2)
                    tropes = sorted(set([trope for characterid in batch_characterids
                                               for trope in characterid_to_tropes[characterid]]))
                    n_tropes_batch_arr.append(len(tropes))
                    n_pos_tropes_batch = 0
                    for characterid in batch_characterids:
                        for trope in tropes:
                            if trope in characterid_to_tropes[characterid]:
                                label = self.characterid_and_trope_to_label[(characterid, trope)]
                                n_pos_tropes_batch += label == 1
                                n_samples += 1
                    frac_pos_tropes_batch_arr.append(n_pos_tropes_batch/len(tropes))
                    character_batches.append({"character-ids": batch_characterids,
                                              "batch-token-ids": batch_token_ids,
                                              "batch-names-idx": batch_names_idx,
                                              "batch-mentions-idx": batch_mentions_idx,
                                              "batch-utterances-idx": batch_utterances_idx,
                                              "tropes": tropes})
                    i = j
            n_labeled_samples = (self.data_df["partition"] == partition.lower()).sum()
            n_missed_samples = n_labeled_samples - n_samples
            print(f"{partition}")
            print(f"{n_characters_with_n_blocks_grt_max_n_blocks} characters had their segments truncated")
            print(f"{n_labeled_samples} labeled samples, {n_missed_samples} samples removed because character did "
                  "not have any mentions or utterances")
            print(f"{n_samples} samples, {len(character_batches)} batches")
            print(f"tropes per batch: avg = {np.mean(n_tropes_batch_arr):.1f} "
                  f"[{min(n_tropes_batch_arr)}, {max(n_tropes_batch_arr)}]")
            print(f"fraction positive samples per batch = {np.mean(frac_pos_tropes_batch_arr):.1f} "
                  f"[{min(frac_pos_tropes_batch_arr):.1f}, {max(frac_pos_tropes_batch_arr):.1f}]")
            character_batches_arr.append(character_batches)
        return character_batches_arr