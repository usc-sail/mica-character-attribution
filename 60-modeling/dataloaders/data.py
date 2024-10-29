"""Functions for data preprocessing"""
import collections
import math
import numpy as np
import pandas as pd
import os
import random
import torch
import tqdm

class Chatter:

    def __init__(self, data_df, character_movie_map_df) -> None:
        data_df = data_df.merge(character_movie_map_df[["character", "component"]].drop_duplicates(),
                                how="left", on="character")
        imdbid_and_charactername_to_characterid = {}
        characterid_to_imdbids = collections.defaultdict(set)
        characterid_and_trope_to_label = {}

        for characterid, imdbid, name in (character_movie_map_df[["character", "imdb-id", "name"]]
                                          .itertuples(index=False, name=None)):
            imdbid_and_charactername_to_characterid[(imdbid, name)] = characterid
            characterid_to_imdbids[characterid].add(imdbid)

        for characterid, trope, tvtropelabel, label in (data_df[["character", "trope", "tvtrope-label", "label"]]
                                                        .itertuples(index=False, name=None)):
            characterid_and_trope_to_label[(characterid, trope)] = label if pd.notna(label) else tvtropelabel

        self.data_df = data_df
        self.map_df = character_movie_map_df
        self.imdbid_and_charactername_to_characterid = imdbid_and_charactername_to_characterid
        self.characterid_to_imdbids = self.sort_dictionary(characterid_to_imdbids)
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
            names_mask_file = os.path.join(movie_dir, "names-mask.pt")
            mentions_mask_file = os.path.join(movie_dir, "mentions-mask.pt")
            utterances_mask_file = os.path.join(movie_dir, "utterances-mask.pt")
            mention_character_ids_file = os.path.join(movie_dir, "mention-character-ids.pt")
            utterance_character_ids_file = os.path.join(movie_dir, "utterance-character-ids.pt")
            characternames = open(characters_file).read().split("\n")
            characterids = [self.imdbid_and_charactername_to_characterid[(imdbid, name)] for name in characternames]
            token_ids = torch.load(token_ids_file, weights_only=True).cuda()
            names_mask = torch.load(names_mask_file, weights_only=True).cuda()
            mentions_mask = torch.load(mentions_mask_file, weights_only=True).cuda()
            utterances_mask = torch.load(utterances_mask_file, weights_only=True).cuda()
            mention_character_ids = torch.load(mention_character_ids_file, weights_only=True).cuda()
            utterance_character_ids = torch.load(utterance_character_ids_file, weights_only=True).cuda()
            imdbid_to_tensors[imdbid] = {"character-ids": characterids,
                                         "token-ids": token_ids,
                                         "names-mask": names_mask,
                                         "mentions-mask": mentions_mask,
                                         "utterances-mask": utterances_mask,
                                         "mentions-character-ids": mention_character_ids,
                                         "utterances-character-ids": utterance_character_ids,
                                         }
        return imdbid_to_tensors

    def batch_components(self, max_ntropes_per_component):
        self.componentid_to_characterids_and_tropes = {}
        componentids_arr = []
        s = 0
        random.seed(12)

        for partition in ["train", "dev", "test"]:
            batchcomponentid_to_characterids_and_tropes = collections.defaultdict(set)
            componentid_to_characterids_and_tropes = {}
            componentids = self.data_df.loc[self.data_df["partition"] == partition, "component"].unique()
            componentid_to_tropes = {}
            component_ntropes = []

            for componentid, cdf in self.data_df[self.data_df["partition"] == partition].groupby("component"):
                componentid_to_characterids_and_tropes[componentid] = set(
                    cdf[["character", "trope"]].itertuples(index=False, name=None))

            for componentid in componentids:
                tropes = set([trope for _, trope in componentid_to_characterids_and_tropes[componentid]])
                componentid_to_tropes[componentid] = sorted(tropes)
                component_ntropes.append(len(tropes))

            component_ntropes = np.array(component_ntropes)
            sortindex = np.argsort(component_ntropes)
            componentids = componentids[sortindex]
            component_ntropes = component_ntropes[sortindex]

            i = 0
            while i < len(componentids):
                if component_ntropes[i] <= max_ntropes_per_component:
                    j = i + 1
                    while j < len(componentids) and component_ntropes[i: j + 1].sum() <= max_ntropes_per_component:
                        j += 1
                    batchcomponentid = "D" + str(s).zfill(4)
                    for componentid in componentids[i: j]:
                        batchcomponentid_to_characterids_and_tropes[batchcomponentid].update(
                            componentid_to_characterids_and_tropes[componentid])
                    s += 1
                    i = j
                else:
                    nsubcomponents = math.ceil(component_ntropes[i]/max_ntropes_per_component)
                    subcomponent_ntropes = math.ceil(component_ntropes[i]/nsubcomponents)
                    tropes = sorted(componentid_to_tropes[componentids[i]])
                    random.shuffle(tropes)
                    for j in range(nsubcomponents):
                        subcomponent_tropes = tropes[j * subcomponent_ntropes: (j + 1) * subcomponent_ntropes]
                        batchcomponentid = "D" + str(s).zfill(4)
                        batchcomponentid_to_characterids_and_tropes[batchcomponentid] = set(
                            [(characterid, trope)
                            for characterid, trope in componentid_to_characterids_and_tropes[componentids[i]]
                                if trope in subcomponent_tropes])
                        s += 1
                    i += 1
            assert (sum([len(characterids_and_tropes)
                        for characterids_and_tropes in batchcomponentid_to_characterids_and_tropes.values()])
                        == (self.data_df["partition"] == partition).sum())

            batchcomponent_ncharacters = []
            batchcomponent_ntropes = []
            batchcomponent_nmovies = []
            batchcomponent_embeddingsizes = []
            batchcomponent_ncharacters_and_tropes = []
            batchcomponent_percentage_positive = []

            for characterids_and_tropes in batchcomponentid_to_characterids_and_tropes.values():
                characterids = set()
                tropes = set()
                imdbids = set()
                npositive = 0
                for characterid, trope in characterids_and_tropes:
                    characterids.add(characterid)
                    tropes.add(trope)
                    imdbids.update(self.characterid_to_imdbids[characterid])
                    npositive += self.characterid_and_trope_to_label[(characterid, trope)] == 1
                batchcomponent_ncharacters.append(len(characterids))
                batchcomponent_ntropes.append(len(tropes))
                batchcomponent_nmovies.append(len(imdbids))
                batchcomponent_embeddingsizes.append(len(characterids) * len(tropes) * len(imdbids))
                batchcomponent_ncharacters_and_tropes.append(len(characterids_and_tropes))
                batchcomponent_percentage_positive.append(100*npositive/len(characterids_and_tropes))
            print(f"{partition}: {len(batchcomponentid_to_characterids_and_tropes)} batches\n"
                    f"\tcharacters ~ {np.mean(batchcomponent_ncharacters):.1f} "
                    f"({np.std(batchcomponent_ncharacters):.2f}) "
                    f"[{min(batchcomponent_ncharacters)}, {max(batchcomponent_ncharacters)}]\n"
                    f"\ttropes ~ {np.mean(batchcomponent_ntropes):.1f} "
                    f"({np.std(batchcomponent_ntropes):.2f}) "
                    f"[{min(batchcomponent_ntropes)}, {max(batchcomponent_ntropes)}]\n"
                    f"\tmovies ~ {np.mean(batchcomponent_nmovies):.1f} "
                    f"({np.std(batchcomponent_nmovies):.2f}) "
                    f"[{min(batchcomponent_nmovies)}, {max(batchcomponent_nmovies)}]\n"
                    f"\tembedding-size ~ {np.mean(batchcomponent_embeddingsizes):.1f} "
                    f"({np.std(batchcomponent_embeddingsizes):.2f}) "
                    f"[{min(batchcomponent_embeddingsizes)}, {max(batchcomponent_embeddingsizes)}]\n"
                    f"\tsamples ~ {np.mean(batchcomponent_ncharacters_and_tropes):.1f} "
                    f"({np.std(batchcomponent_ncharacters_and_tropes):.2f}) "
                    f"[{min(batchcomponent_ncharacters_and_tropes)}, {max(batchcomponent_ncharacters_and_tropes)}]\n"
                    f"\t%+ve/batch ~ {np.mean(batchcomponent_percentage_positive):.1f} "
                    f"({np.std(batchcomponent_percentage_positive):.2f}) "
                    f"[{min(batchcomponent_percentage_positive):.1f}, {max(batchcomponent_percentage_positive):.1f}]\n"
                    )
            self.componentid_to_characterids_and_tropes.update(dict(batchcomponentid_to_characterids_and_tropes))
            componentids_arr.append(sorted(batchcomponentid_to_characterids_and_tropes.keys()))

        self.componentid_to_characterids_and_tropes = self.sort_dictionary(self.componentid_to_characterids_and_tropes)
        self.train_componentids, self.dev_componentids, self.test_componentids = componentids_arr