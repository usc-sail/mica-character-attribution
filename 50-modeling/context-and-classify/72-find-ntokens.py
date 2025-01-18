"""Find number of tokens"""
import datadirs

from absl import app
from absl import flags
import json
import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer

FLAGS = flags.FLAGS
flags.DEFINE_enum("source", default="all", enum_values=["all", "desc", "utter"], help="character data source in story")
flags.DEFINE_enum("context", default="0", enum_values=["0", "1", "5", "10", "20"],
                  help="number of neighboring segments")

def create_text(definition, name, contexts, tokenizer):
    return definition + tokenizer.bos_token + name + tokenizer.bos_token + tokenizer.bos_token.join(contexts)

def find_ntokens(_):
    """Find number of tokens"""
    # get filepaths
    context_file = os.path.join(datadirs.datadir,
                                f"70-classification/contexts/{FLAGS.source}-context-{FLAGS.context}.json")
    label_file = os.path.join(datadirs.datadir, "CHATTER/chatter.csv")
    map_file = os.path.join(datadirs.datadir, "CHATTER/character-movie-map.csv")
    tropes_file = os.path.join(datadirs.datadir, "CHATTER/tropes.csv")
    ntokens_file = os.path.join(datadirs.datadir,
                                f"70-classification/ntokens/{FLAGS.source}-context-{FLAGS.context}-llama3.csv")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # load contexts
    characterid_to_contexts = json.load(open(context_file))
    characterids = characterid_to_contexts.keys()

    # find tropes for which we have labels for each character
    label_df = pd.read_csv(label_file, index_col=None)
    characterid_to_tropes = {}
    for characterid, character_df in label_df.groupby("character"):
        characterid_to_tropes[characterid] = set(character_df.loc[character_df["partition"].notna(), "trope"])

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
    textids = []
    texts = []
    for characterid in characterids:
        contexts = characterid_to_contexts[characterid]
        name = characterid_to_name[characterid]
        for trope in characterid_to_tropes[characterid]:
            definition = trope_to_definition[trope]
            text = create_text(definition, name, contexts, tokenizer)
            textid = [characterid, trope]
            texts.append(text)
            textids.append(textid)

    # tokenizing
    tokenids_arr = tokenizer(texts).input_ids
    tokens_arr = [text.split() for text in texts]
    ntokens = [len(tokenids) for tokenids in tokenids_arr]
    nwtokens = [len(tokens) for tokens in tokens_arr]

    # save output
    ntokens_df = pd.DataFrame(columns=["character", "trope", "ntokens", "nwtokens"])
    ntokens_df[["character", "trope"]] = textids
    ntokens_df["ntokens"] = ntokens
    ntokens_df["nwtokens"] = nwtokens
    ntokens_df.to_csv(ntokens_file, index=False)

if __name__ == '__main__':
    app.run(find_ntokens)