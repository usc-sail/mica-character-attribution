"""Tokenize texts for classification"""
import datadirs

from absl import app
from absl import flags
import json
import math
import numpy as np
import os
import pandas as pd
import torch
import tqdm
from transformers import AutoTokenizer

FLAGS = flags.FLAGS
flags.DEFINE_enum("source", default="all", enum_values=["all", "desc", "utter"], help="character data source in story")
flags.DEFINE_enum("context", default="0", enum_values=["0", "1", "5", "10", "20"],
                  help="number of neighboring segments")
flags.DEFINE_integer("loglen", default=13, lower_bound=13, upper_bound=17, help="base2 log of sequence length")

def create_text(definition, name, contexts, tokenizer):
    return definition + tokenizer.bos_token + name + tokenizer.bos_token + tokenizer.bos_token.join(contexts)

def tokenize(_):
    """Find number of tokens"""
    # get filepaths
    length = 1 << FLAGS.loglen
    Ksize = 1 << (FLAGS.loglen - 10)
    context_file = os.path.join(datadirs.datadir,
                                f"70-classification/contexts/{FLAGS.source}-context-{FLAGS.context}.json")
    label_file = os.path.join(datadirs.datadir, "CHATTER/chatter.csv")
    map_file = os.path.join(datadirs.datadir, "CHATTER/character-movie-map.csv")
    tropes_file = os.path.join(datadirs.datadir, "CHATTER/tropes.csv")
    output_dir = os.path.join(datadirs.datadir,
                              f"70-classification/data/{FLAGS.source}-context-{FLAGS.context}-llama3-{Ksize}K")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # load contexts
    characterid_to_contexts = json.load(open(context_file))
    characterids = characterid_to_contexts.keys()

    # find train, dev, and test ids and the labels
    label_df = pd.read_csv(label_file, index_col=None)
    id_to_label = {}
    for _, row in label_df[label_df.partition.notna()].iterrows():
        _id = (row.character, row.trope)
        if row.partition == "test":
            id_to_label[_id] = [row.label, "test"]
        else:
            id_to_label[_id] = [row["tvtrope-label"], row.partition]

    # find character name
    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    characterid_to_name = {}
    for characterid, character_df in map_df.groupby("character"):
        names = character_df["name"].unique().tolist()
        if len(names) == 1:
            characterid_to_name[characterid] = names[0]
        else:
            name_sizes = [len(name) for name in names]
            i = np.argmax(name_sizes)
            characterid_to_name[characterid] = names[i]

    # find trope definitions
    tropes_df = pd.read_csv(tropes_file, index_col=None, dtype=str)
    trope_to_definition = {tup.trope: tup.definition for tup in tropes_df.itertuples(index=False)}

    # create input texts
    texts, labels = [], []
    for _id, label in id_to_label.items():
        characterid, trope = _id
        name = characterid_to_name[characterid]
        definition = trope_to_definition[trope]
        if characterid in characterids:
            contexts = characterid_to_contexts[characterid]
            text = create_text(definition, name, contexts, tokenizer)
            texts.append(text)
            labels.append([characterid, trope] + label)

    # tokenization
    token_ids_arr = []
    n_batches = math.ceil(len(texts)/1000)
    for i in tqdm.trange(n_batches, desc="tokenization"):
        batch_texts = texts[i*1000: (i + 1)*1000]
        token_ids = tokenizer(batch_texts, padding="max_length", truncation=True, padding_side="right",
                              add_special_tokens=True, return_tensors="pt", max_length=length).input_ids
        token_ids_arr.append(token_ids)
    token_ids = torch.cat(token_ids_arr, dim=0)

    # create labels dataframe
    labels_df = pd.DataFrame(labels, columns=["character", "trope", "label", "partition"])

    # write tokens and labels
    os.makedirs(output_dir, exist_ok=True)
    tokens_file = os.path.join(output_dir, "tokens.pt")
    texts_file = os.path.join(output_dir, "texts.json")
    labels_file = os.path.join(output_dir, "labels.csv")
    torch.save(token_ids, tokens_file)
    json.dump(texts_file, open(texts_file, "w"))
    labels_df.to_csv(labels_file, index=False)

if __name__ == '__main__':
    app.run(tokenize)