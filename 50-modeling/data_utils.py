"""Utility functions to load data for the CHATTER, PERSONET and STORY2PERSONALITY datasets in a standard format

Each data sample contains the following keys: key, docid, character, attribute-name, attribute-definition, text, label,
partition
"""
import collections
from datasets import Dataset
import jsonlines
import numpy as np
import os
import pandas as pd
import pickle
import random
import socket
import string
import tqdm
from transformers import AutoTokenizer
from typing import List, Dict, Union, Tuple

HOST = socket.gethostname()
if HOST == "redondo":
    # lab server
    DATADIR = "/data1/sbaruah/mica-character-attribution"
elif HOST.endswith("hpc.usc.edu"):
    # university HPC compute
    DATADIR = "/scratch1/sbaruah/mica-character-attribution"
else:
    # AWS EC2
    DATADIR = "/home/ubuntu/data/mica-character-attribution"

TEMPLATE = ("Given the definition of a character attribute or trope, the name of a character, and a story or segments "
            "of a story where the character appears, speaks or is mentioned, answer 'yes' or 'no' if the character "
            "portrays or is associated with the attribute or trope in the story.\n\nATTRIBUTE: $ATTRIBUTE$"
            "\n\nCHARACTER: $CHARACTER$\n\nSTORY: $STORY$. \n\n ANSWER: $ANSWER$")
ANSWER_TEMPLATE = " \n\n ANSWER:"
CHARACTER_TOKEN = "[CHARACTER]"
ATTRIBUTE_TOKEN = "[ATTRIBUTE]"
CONTEXT_TOKEN = "[CONTEXT]"

def load_extracts(extracts_file, anonymize=False):
    """Load extracts data"""
    with jsonlines.open(extracts_file) as reader:
        data = list(reader)
    processed_data = []
    for obj in data:
        processed_data.append({"key": f"{obj['character']}-{obj['trope']}",
                               "docid": obj["imdbid"],
                               "character": "CHARACTER" + obj["character"][1:] if anonymize else obj["name"],
                               "attribute-name": obj["trope"],
                               "attribute-definition": obj["definition"],
                               "text": obj["text"],
                               "label": obj["label"],
                               "partition": obj["partition"]})
    return processed_data

def load_contexts(contexts_file, anonymize=False):
    """Load contexts data"""
    with jsonlines.open(contexts_file) as reader:
        data = list(reader)
    processed_data = []
    if "partition" in data[0]:
        for obj in data:
            processed_data.append({"key": f"{obj['character']}-{obj['trope']}",
                                   "docid": obj["imdbid"],
                                   "character": "CHARACTER" + obj["character"][1:] if anonymize else obj["name"],
                                   "attribute-name": obj["trope"],
                                   "attribute-definition": obj["definition"],
                                   "text": obj["text"],
                                   "label": obj["label"],
                                   "partition": obj["partition"]})
    else:
        label_file = os.path.join(DATADIR, "CHATTER/chatter.csv")
        tropes_file = os.path.join(DATADIR, "CHATTER/tropes.csv")
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
            label = int(label)
            for i in characterid_to_ixs[characterid]:
                obj = data[i]
                processed_data.append({"key": f"{characterid}-{trope}",
                                       "docid": obj["imdbid"],
                                       "character": "CHARACTER" + obj["character"][1:] if anonymize else obj["name"],
                                       "attribute-name": trope,
                                       "attribute-definition": definition,
                                       "text": obj["text"],
                                       "label": label,
                                       "partition": partition})
    return processed_data

def load_personet():
    """Load personet data"""
    personet_dir = os.path.join(DATADIR, "PERSONET")
    with jsonlines.open(os.path.join(personet_dir, "test.jsonl")) as reader:
        test_data = list(reader)
    for obj in test_data:
        obj["partition"] = "test"
    with jsonlines.open(os.path.join(personet_dir, "train.jsonl")) as reader:
        train_data = list(reader)
    for obj in train_data:
        obj["partition"] = "train"
    with jsonlines.open(os.path.join(personet_dir, "dev.jsonl")) as reader:
        dev_data = list(reader)
    for obj in dev_data:
        obj["partition"] = "dev"
    processed_data = []
    for obj in test_data + train_data + dev_data:
        traits = obj["options"]
        answer = ord(obj["answer"][1]) - ord("a")
        text = "\n".join([obj["history"], obj["snippet_former_context"], obj["snippet_underlined"],
                          obj["snippet_post_context"]])
        key = obj["key"] if "key" in obj else "".join(random.choice(string.ascii_letters + string.digits)
                                                      for _ in range(10))
        processed_obj = {"key": key,
                         "docid": obj["book_name"],
                         "character": obj["character"],
                         "text": text,
                         "partition": obj["partition"]}
        if obj["partition"] in ["test", "dev"]:
            for i, trait in enumerate(traits):
                processed_data.append({**processed_obj,
                                       "attribute-name": trait,
                                       "attribute-definition": trait,
                                       "label": int(answer == i)})
        else:
            wrong_answer = random.choice([i for i in range(len(traits)) if i != answer])
            positive_processed_obj = {**processed_obj,
                                      "attribute-name": traits[answer],
                                      "attribute-definition": traits[answer],
                                      "label": 1}
            negative_processed_obj = {**processed_obj,
                                      "attribute-name": traits[wrong_answer],
                                      "attribute-definition": traits[wrong_answer],
                                      "label": 0}
            processed_data.extend([positive_processed_obj, negative_processed_obj])
    return processed_data

def load_story2personality():
    """Load story2personality data"""
    filepath = os.path.join(DATADIR, "STORY2PERSONALITY/BERT.tok.pkl")
    definition_filepath = os.path.join(DATADIR, "STORY2PERSONALITY/personality-definitions.txt")
    story2personality = pickle.load(open(filepath, mode="rb"))
    personality2definition = {}
    personality2name = {"E": "Extraversion",
                        "I": "Introversion",
                        "S": "Sensing",
                        "N": "Intuition",
                        "T": "Thinking",
                        "F": "Feeling",
                        "J": "Judging",
                        "P": "Perceiving"}
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
                                       "attribute-name": personality2name[px], 
                                       "attribute-definition": dx,
                                       "text": text,
                                       "label": lx,
                                       "partition": "test"})
                processed_data.append({"key": f"{obj['id']}-{px}/{py}",
                                       "docid": obj["subcategory"],
                                       "character": obj["mbti_profile"],
                                       "attribute-name": personality2name[py],
                                       "attribute-definition": dy,
                                       "text": text,
                                       "label": ly,
                                       "partition": "test"})
    return processed_data

def create_dataset(tokenizer: AutoTokenizer,
                   data: List[Dict[str, Union[str, int]]],
                   instrtune = False,
                   batch_size = 256,
                   instr_max_seqlen = 1024,
                   disable_progress_bar = False) -> Tuple[Dataset, pd.DataFrame]:
    """Create Dataset object and evaluation dataframe from data array"""
    texts = []
    rows = []
    for obj in data:
        if instrtune:
            text = (TEMPLATE
                    .replace("$ANSWER$", "yes" if obj["label"] == 1 else "no")
                    .replace("$CHARACTER$", obj["character"])
                    .replace("$ATTRIBUTE$", obj["attribute-definition"])
                    .replace("$STORY$", obj["text"])
                    )
        else:
            text = (f"{ATTRIBUTE_TOKEN}{obj['attribute-definition']}{CHARACTER_TOKEN}{obj['character']}{CONTEXT_TOKEN}"
                    f"{obj['text']}")
        row = [obj["key"], obj["attribute-name"], obj["label"]]
        rows.append(row)
        texts.append(text)
    df = pd.DataFrame(rows, columns=["key", "attribute", "label"])
    input_ids, attention_mask = [], []
    n_batches = int(np.ceil(len(texts)/batch_size))
    for i in tqdm.trange(n_batches, desc="tokenization", disable=disable_progress_bar):
        batch_texts = texts[batch_size * i: batch_size * (i + 1)]
        encoding = tokenizer(batch_texts,
                             padding=False,
                             add_special_tokens=True,
                             return_overflowing_tokens=False,
                             return_length=False)
        input_ids.extend(encoding["input_ids"])
        attention_mask.extend(encoding["attention_mask"])
    if instrtune:
        yes_answer_sp_mask = tokenizer(ANSWER_TEMPLATE + " yes",
                                       add_special_tokens=True,
                                       return_special_tokens_mask=True)["special_tokens_mask"]
        no_answer_sp_mask = tokenizer(ANSWER_TEMPLATE + " no",
                                       add_special_tokens=True,
                                       return_special_tokens_mask=True)["special_tokens_mask"]
        is_first_token_sp = yes_answer_sp_mask[0] == 1
        is_last_token_sp = yes_answer_sp_mask[-1] == 1
        yes_answer_size = len(yes_answer_sp_mask) - is_first_token_sp - is_last_token_sp
        no_answer_size = len(no_answer_sp_mask) - is_first_token_sp - is_last_token_sp
        for i in tqdm.trange(len(input_ids), desc="truncation", disable=disable_progress_bar):
            if len(input_ids[i]) > instr_max_seqlen:
                if data[i]["label"] == 1:
                    start = -is_last_token_sp - yes_answer_size - (len(input_ids[i]) - instr_max_seqlen)
                    end = -is_last_token_sp - yes_answer_size
                else:
                    start = -is_last_token_sp - no_answer_size - (len(input_ids[i]) - instr_max_seqlen)
                    end = -is_last_token_sp - no_answer_size
                input_ids[i] = input_ids[i][:start] + input_ids[i][end:]
                attention_mask[i] = attention_mask[i][:start] + attention_mask[i][end:]
        dataset = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})
    else:
        dataset = Dataset.from_dict({"input_ids": input_ids,
                                     "attention_mask": attention_mask,
                                     "labels": df["label"].tolist()})
    return dataset, df

if __name__ == '__main__':
    extracts_file = os.path.join(DATADIR, "50-modeling/extracts/Llama-3.1-8B-Instruct-1536-train.jsonl")
    contexts_file1 = os.path.join(DATADIR, "50-modeling/contexts/25P-1000C-random.jsonl")
    contexts_file2 = os.path.join(DATADIR,
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