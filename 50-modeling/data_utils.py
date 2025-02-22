"""Utility functions to load data for the CHATTER, PERSONET and STORY2PERSONALITY datasets"""
import datadirs

import collections
import jsonlines
import os
import pandas as pd
import pickle
import random
import string

def load_extracts(extracts_file):
    """Load extracts data"""
    with jsonlines.open(extracts_file) as reader:
        data = list(reader)
    processed_data = []
    for obj in data:
        processed_data.append({"key": f"{obj['character']}-{obj['trope']}",
                               "docid": obj["imdbid"],
                               "character": obj["name"],
                               "attribute": obj["definition"],
                               "text": obj["text"],
                               "label": obj["label"],
                               "partition": obj["partition"]})
    return processed_data

def load_contexts(contexts_file):
    """Load contexts data"""
    with jsonlines.open(contexts_file) as reader:
        data = list(reader)
    processed_data = []
    if "partition" in data[0]:
        for obj in data:
            processed_data.append({"key": f"{obj['character']}-{obj['trope']}",
                                   "docid": obj["imdbid"],
                                   "character": obj["name"],
                                   "attribute": obj["definition"],
                                   "text": obj["text"],
                                   "label": obj["label"],
                                   "partition": obj["partition"]})
    else:
        label_file = os.path.join(datadirs.datadir, "CHATTER/chatter.csv")
        tropes_file = os.path.join(datadirs.datadir, "CHATTER/tropes.csv")
        label_df = pd.read_csv(label_file, index_col=None)
        tropes_df = pd.read_csv(tropes_file, index_col=None)
        characterid_to_ixs = collections.defaultdict(list)
        trope_to_definition = {}
        for i, obj in enumerate(data):
            characterid = obj["character"]
            characterid_to_ixs[characterid].append(i)
        for _, row in tropes_df.iterrows():
            trope_to_definition[row["trope"]] = row["summary"]
        for _, row in label_df[label_df["partition"].notna()].iterrows():
            characterid, trope, partition = row["character"], row["trope"], row["partition"]
            definition = trope_to_definition[trope]
            label = row["label"] if partition == "test" else row["tvtrope-label"]
            for i in characterid_to_ixs[characterid]:
                obj = data[i]
                processed_data.append({"key": f"{characterid}-{trope}",
                                       "docid": obj["imdbid"],
                                       "character": obj["name"],
                                       "attribute": definition,
                                       "text": obj["text"],
                                       "label": label,
                                       "partition": partition})
    return processed_data

def load_personet(test=False):
    """Load personet data"""
    personet_dir = os.path.join(datadirs.datadir, "PERSONET")
    with jsonlines.open(os.path.join(personet_dir, "test.jsonl")) as reader:
        data = list(reader)
    for obj in data:
        obj["partition"] = "test"
    if not test:
        with jsonlines.open(os.path.join(personet_dir, "train.jsonl")) as reader:
            train_data = list(reader)
        for obj in train_data:
            obj["partition"] = "train"
        with jsonlines.open(os.path.join(personet_dir, "dev.jsonl")) as reader:
            dev_data = list(reader)
        for obj in dev_data:
            obj["partition"] = "dev"
        data += train_data + dev_data
    processed_data = []
    for obj in data:
        traits = obj["options"]
        answer = ord(obj["answer"][1]) - ord("a")
        text = "\n".join([obj["history"], obj["snippet_former_context"], obj["snippet_underlined"],
                          obj["snippet_post_context"]])
        key = obj["key"] if "key" in obj else "".join(random.choice(string.ascii_letters + string.digits)
                                                      for _ in range(10))
        for i, trait in enumerate(traits):
            processed_data.append({"key": key,
                                   "docid": obj["book_name"],
                                   "character": obj["character"],
                                   "attribute": trait,
                                   "text": text,
                                   "label": int(answer == i),
                                   "partition": obj["partition"]})
    return processed_data

def load_story2personality():
    """Load story2personality data"""
    filepath = os.path.join(datadirs.datadir, "STORY2PERSONALITY/BERT.tok.pkl")
    definition_filepath = os.path.join(datadirs.datadir, "STORY2PERSONALITY/personality-definitions.txt")
    story2personality = pickle.load(open(filepath, mode="rb"))
    personality2definition = {}
    with open(definition_filepath) as fr:
        lines = fr.read().strip().split("\n")
    for i in range(0, len(lines), 2):
        personality2definition[lines[i].strip()] = lines[i + 1].strip()
    processed_data = []
    for obj in story2personality:
        utterances = "\n".join(list(set(obj["dialog_text"])))
        mentions = "\n".join(list(set([item[-1] for item in obj["scene_text"]])))
        text = f"UTTERANCES:\n{utterances}\n\nMENTIONS:\n{mentions}"
        text = text.strip()
        for px, py in ["EI", "SN", "TF", "JP"]:
            nx, ny = obj[px], obj[py]
            dx, dy = personality2definition[px], personality2definition[py]
            if pd.notna([nx, ny]).any():
                if pd.notna(nx):
                    lx, ly = 1, 0
                else:
                    lx, ly = 0, 1
                processed_data.append({"key": f"{obj['id']}-{px}/{py}",
                                       "docid": obj["subcategory"],
                                       "character": obj["mbti_profile"],
                                       "attribute": dx,
                                       "text": text,
                                       "label": lx,
                                       "partition": "test"})
                processed_data.append({"key": f"{obj['id']}-{px}/{py}",
                                       "docid": obj["subcategory"],
                                       "character": obj["mbti_profile"],
                                       "attribute": dy,
                                       "text": text,
                                       "label": ly,
                                       "partition": "test"})
    return processed_data

if __name__ == '__main__':
    extracts_file = os.path.join(datadirs.datadir, "50-modeling/extracts/Llama-3.1-8B-Instruct-1536-train.jsonl")
    contexts_file1 = os.path.join(datadirs.datadir, "50-modeling/contexts/25P-1000C-random.jsonl")
    contexts_file2 = os.path.join(datadirs.datadir,
                                  "50-modeling/contexts/25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl")
    print("testing...")
    print("load extracts")
    data = load_extracts(extracts_file)
    print(f"{len(data)} samples loaded")
    print("load contexts (no partition)")
    data = load_contexts(contexts_file1)
    print(f"{len(data)} samples loaded")
    print("load contexts (partition)")
    data = load_contexts(contexts_file2)
    print(f"{len(data)} samples loaded")
    print("load personet")
    data = load_personet()
    print(f"{len(data)} samples loaded")
    print("load story2personality")
    data = load_story2personality()
    print(f"{len(data)} samples loaded")
    print("done testing...")