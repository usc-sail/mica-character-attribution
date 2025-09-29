"""Load data for the CHATTER and PERSONET datasets in a standard format

A CHATTER data sample contains the following keys: key, docid, character, attribute-name, attribute-definition, text, label, partition

A PERSONET data sample contains the following keys: key, docid, character, attributes, text, label, partition
"""
import collections
from datasets import Dataset
import jsonlines
import numpy as np
import os
import pandas as pd
import socket
import tqdm
from transformers import AutoTokenizer
from typing import List, Dict, Union, Tuple, Literal

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

CHATTER_TEMPLATE = ("Given the definition of a character attribute or trope, the name of a character, and a story or "
                    "segments of a story where the character appears, speaks or is mentioned, answer 'yes' or 'no' if "
                    "the character portrays or is associated with the attribute or trope in the story.\n\n"
                    "ATTRIBUTE: $ATTRIBUTE$\n\nCHARACTER: $CHARACTER$\n\nSTORY: $STORY$. \n\n ANSWER: $ANSWER$")
PERSONET_TEMPLATE = ("Given some character attributes or traits, the name of a character, "
                     "and an excerpt from a story where the character appears, speaks or is mentioned, "
                     "pick the index of the attribute (index starts from 1) most strongly portrayed by "
                     "or associated with the character.\n\n"
                     "Do not return the attribute, just its index. "
                     "For example, if the attributes are \"strong\", \"kind\", \"gentle\", and \"weak\", "
                     "and the character most strongly portrays being kind, then return \"2\".\n\n"
                     "ATTRIBUTES:\n1. $ATTRIBUTE1$\n2. $ATTRIBUTE2$\n3. $ATTRIBUTE3$\n4. $ATTRIBUTE4$\n"
                     "5. $ATTRIBUTE5$\n\nCHARACTER: $CHARACTER$\n\nSTORY: $STORY$. \n\n ANSWER: $ANSWER$")
ANSWER_TEMPLATE = " \n\n ANSWER:"
NCLASSES = 5
CHATTER_CONTEXTS_WORD_SIZE_TO_SFT_SEQLEN = {250: 550, 500: 1000, 1000: 1800, 1500: 2700, 2000: 3500}
PERSONET_SFT_SEQLEN = 1800
CHATTER_SEGMENTS_SEQLEN = 14000

def get_dataset_names():
    dataset_names = []
    for partition in ["train", "dev", "test"]:
        dataset_names.append(f"personet-{partition}")
        for preprocess in ["original", "anonymized"]:
            dataset_names.append(f"chatter-segments-{preprocess}-{partition}")
            for truncation_strategy in ["first", "semantic"]:
                for size in [250, 500, 1000, 1500, 2000]:
                    dataset_names.append(f"chatter-contexts-{preprocess}-{truncation_strategy}-{size}-{partition}")
    return dataset_names

def load_chatter_contexts(truncation_strategy: Literal["first", "semantic"] = "first",
                          size_in_words: Literal[250, 500, 1000, 1500, 2000] = 2000,
                          anonymize = False,
                          partitions = ["train", "dev", "test"]):
    """Load chatter contexts data"""
    # get contexts file
    contexts_dir = "anonymized-contexts" if anonymize else "contexts"
    if truncation_strategy == "first":
        contexts_file = os.path.join(DATADIR, "CHATTER", contexts_dir, f"25P-{size_in_words}C-first.jsonl")
    else:
        contexts_file = os.path.join(DATADIR, "CHATTER", contexts_dir,
                                     f"25P-{size_in_words}C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl")

    # read contexts data
    with jsonlines.open(contexts_file) as reader:
        data = list(reader)

    # process contexts data
    processed_data = []
    if truncation_strategy == "semantic":
        for obj in data:
            if obj["partition"] in partitions:
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
        for _, row in label_df[label_df["partition"].isin(partitions)].iterrows():
            characterid, trope, partition = row["character"], row["trope"], row["partition"]
            definition = trope_to_definition[trope]
            label = row["label"] if partition == "test" else row["tvtrope-label"]
            label = int(label)
            for i in characterid_to_ixs[characterid]:
                obj = data[i]
                processed_data.append({"key": f"{characterid}-{trope}",
                                    "docid": obj["imdbid"],
                                    "character": "CHARACTER"+obj["character"][1:] if anonymize else obj["name"],
                                    "attribute-name": trope,
                                    "attribute-definition": definition,
                                    "text": obj["text"],
                                    "label": label,
                                    "partition": partition})
    return processed_data

def load_chatter_segments(anonymize = False, partitions = ["train", "dev", "test"]):
    """Load chatter segments data"""
    # get file paths
    segments_dir = os.path.join(DATADIR, "CHATTER", "anonymized-segments" if anonymize else "segments")
    map_file = os.path.join(DATADIR, "CHATTER/character-movie-map.csv")
    label_file = os.path.join(DATADIR, "CHATTER/chatter.csv")
    tropes_file = os.path.join(DATADIR, "CHATTER/tropes.csv")

    # read dataframes
    label_df = pd.read_csv(label_file, index_col=None)
    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    tropes_df = pd.read_csv(tropes_file, index_col=None)

    # read segments and character & movie mapping data
    characterid_and_imdbid_to_segment = {}
    characterid_and_imdbid_to_name = {}
    characterid_to_imdbids = {}
    trope_to_definition = {}
    for _, row in tropes_df.iterrows():
        trope_to_definition[row.trope] = row.summary
    for characterid, character_df in tqdm.tqdm(map_df.groupby("character"),
                                               total=len(map_df["character"].unique()),
                                               unit="character",
                                               desc="reading segments"):
        characterid_to_imdbids[characterid] = character_df["imdb-id"].tolist()
        for imdbid, name in character_df[["imdb-id", "name"]].itertuples(index=False, name=None):
            characterid_and_imdbid_to_name[(characterid, imdbid)] = name
            segment_file = os.path.join(segments_dir, f"{characterid}-{imdbid}.txt")
            if os.path.exists(segment_file):
                characterid_and_imdbid_to_segment[(characterid, imdbid)] = open(segment_file).read().strip()

    # process data
    processed_data = []
    for _, row in label_df[label_df["partition"].isin(partitions)].iterrows():
        characterid, trope, partition = row["character"], row["trope"], row["partition"]
        definition = trope_to_definition[trope]
        label = row["label"] if partition == "test" else row["tvtrope-label"]
        label = int(label)
        for imdbid in characterid_to_imdbids[characterid]:
            if (characterid, imdbid) in characterid_and_imdbid_to_segment:
                name = ("CHARACTER" + characterid[1:]
                        if anonymize else characterid_and_imdbid_to_name[(characterid, imdbid)])
                processed_data.append({"key": f"{characterid}-{trope}",
                                       "docid": imdbid,
                                       "character": name,
                                       "attribute-name": trope,
                                       "attribute-definition": definition,
                                       "text": characterid_and_imdbid_to_segment[(characterid, imdbid)],
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
        key = f"{obj['character']}-{traits[answer]}"
        processed_obj = {"key": key,
                         "docid": obj["book_name"],
                         "character": obj["character"],
                         "attributes": traits,
                         "text": text,
                         "label": answer,
                         "partition": obj["partition"]}
        processed_data.append(processed_obj)
    return processed_data

def create_sft_dataset(data: List[Dict[str, Union[str, int]]],
                       tokenizer: AutoTokenizer,
                       dataset_name: Literal["chatter-contexts", "chatter-segments", "personet"] = "chatter-contexts",
                       chatter_contexts_size_in_words: Literal[250, 500, 1000, 1500, 2000] = 2000,
                       tokenization_batch_size = 4096,
                       disable_progress_bar = False) -> Tuple[Dataset, pd.DataFrame]:
    """Create SFT Dataset and evaluation dataframe from data array"""
    texts = []
    rows = []
    input_ids, attention_mask = [], []

    # create the texts for SFT, and the dataframe rows for saving predictions
    for obj in data:
        if dataset_name == "chatter-contexts" or dataset_name == "chatter-segments":
            text = (CHATTER_TEMPLATE
                    .replace("$ANSWER$", "yes" if obj["label"] == 1 else "no")
                    .replace("$CHARACTER$", obj["character"])
                    .replace("$ATTRIBUTE", obj["attribute-definition"])
                    .replace("$STORY$", obj["text"]))
            row = [obj["key"], obj["character"], obj["attribute-name"], obj["label"]]
        else:
            text = (PERSONET_TEMPLATE
                    .replace("$ANSWER$", str(obj["label"] + 1))
                    .replace("$CHARACTER$", obj["character"])
                    .replace("$STORY$", obj["text"]))
            for i in range(NCLASSES):
                text = text.replace(f"$ATTRIBUTE{i + 1}$", obj["attributes"][i])
            row = [obj["key"], obj["character"]] + obj["attributes"] + [obj["label"] + 1]
        texts.append(text)
        rows.append(row)
    
    # create the dataframe for saving predictions
    if dataset_name == "chatter-contexts" or dataset_name == "chatter-segments":
        df = pd.DataFrame(rows, columns=["key", "character", "attribute", "label"])
    else:
        df = pd.DataFrame(rows, columns=["key", "character"] + [f"attribute-{i + 1}" for i in range(NCLASSES)]
                          + ["label"])

    # tokenize the texts
    n_batches = int(np.ceil(len(texts)/tokenization_batch_size))
    for i in tqdm.trange(n_batches, desc=f"{dataset_name} tokenization", disable=disable_progress_bar):
        batch_texts = texts[tokenization_batch_size * i: tokenization_batch_size * (i + 1)]
        encoding = tokenizer(batch_texts,
                             padding=False,
                             add_special_tokens=True,
                             return_overflowing_tokens=False,
                             return_length=False)
        input_ids.extend(encoding["input_ids"])
        attention_mask.extend(encoding["attention_mask"])

    # find size of completions
    chatter_yes_answer_sp_mask = tokenizer(ANSWER_TEMPLATE + " yes",
                                           add_special_tokens=True,
                                           return_special_tokens_mask=True)["special_tokens_mask"]
    chatter_no_answer_sp_mask = tokenizer(ANSWER_TEMPLATE + " no",
                                          add_special_tokens=True,
                                          return_special_tokens_mask=True)["special_tokens_mask"]
    is_first_token_sp = chatter_yes_answer_sp_mask[0] == 1
    is_last_token_sp = chatter_yes_answer_sp_mask[-1] == 1
    chatter_yes_answer_size = len(chatter_yes_answer_sp_mask) - is_first_token_sp - is_last_token_sp
    chatter_no_answer_size = len(chatter_no_answer_sp_mask) - is_first_token_sp - is_last_token_sp
    personet_answer_sizes = []
    for i in range(NCLASSES):
        sp_mask = tokenizer(ANSWER_TEMPLATE + f" {i + 1}",
                            add_special_tokens=True,
                            return_special_tokens_mask=True)["special_tokens_mask"]
        answer_size = len(sp_mask) - is_first_token_sp - is_last_token_sp
        personet_answer_sizes.append(answer_size)

    # truncate to instruction sequence length
    if dataset_name == "chatter-contexts":
        seqlen = CHATTER_CONTEXTS_WORD_SIZE_TO_SFT_SEQLEN[chatter_contexts_size_in_words]
    elif dataset_name == "chatter-segments":
        seqlen = CHATTER_SEGMENTS_SEQLEN
    else:
        seqlen = PERSONET_SFT_SEQLEN
    for i in tqdm.trange(len(input_ids), desc=f"{dataset_name} truncation", disable=disable_progress_bar):
        if len(input_ids[i]) > seqlen:
            if dataset_name == "chatter-contexts" or dataset_name == "chatter-segments":
                if data[i]["label"] == 1:
                    start = -is_last_token_sp - chatter_yes_answer_size - (len(input_ids[i]) - seqlen)
                    end = -is_last_token_sp - chatter_yes_answer_size
                else:
                    start = -is_last_token_sp - chatter_no_answer_size - (len(input_ids[i]) - seqlen)
                    end = -is_last_token_sp - chatter_no_answer_size
            else:
                start = -is_last_token_sp - personet_answer_sizes[data[i]["label"]] - (len(input_ids[i]) - seqlen)
                end = -is_last_token_sp - personet_answer_sizes[data[i]["label"]]
            input_ids[i] = input_ids[i][:start] + input_ids[i][end:]
            attention_mask[i] = attention_mask[i][:start] + attention_mask[i][end:]

    # create dataset
    dataset = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})
    return dataset, df

def create_chr_dataset(data: List[Dict[str, Union[str, int]]],
                       tokenizer: AutoTokenizer,
                       dataset_name: Literal["chatter", "personet"] = "chatter",
                       tokenization_batch_size = 4096,
                       disable_progress_bar = False) -> Tuple[Dataset, pd.DataFrame]:
    """Create Character Representations Dataset and evaluation dataframe from data array"""
    text_pairs = []
    rows = []
    input_ids, attention_mask = [], []

    # create the text pairs for the Character Representations and dataframe rows for saving predictions
    for obj in data:
        text_pair = [obj["character"], obj["text"]]
        if dataset_name == "chatter":
            row = [obj["key"], obj["character"], obj["attribute-name"], obj["label"]]
        else:
            row = [obj["key"], obj["character"]] + obj["attributes"] + [obj["label"] + 1]
        text_pairs.append(text_pair)
        rows.append(row)

    # create the dataframe for saving predictions
    if dataset_name == "chatter":
        df = pd.DataFrame(rows, columns=["key", "character", "attribute", "label"])
    else:
        df = pd.DataFrame(rows, columns=["key", "character"] + [f"attribute-{i + 1}" for i in range(NCLASSES)]
                          + ["label"])

    # tokenize the text pairs
    n_batches = int(np.ceil(len(text_pairs)/tokenization_batch_size))
    for i in tqdm.trange(n_batches, desc=f"{dataset_name} tokenization", disable=disable_progress_bar):
        batch_text_pairs = text_pairs[tokenization_batch_size * i: tokenization_batch_size * (i + 1)]
        encoding = tokenizer(batch_text_pairs,
                             padding=False,
                             add_special_tokens=True,
                             return_overflowing_tokens=False,
                             return_length=False)
        input_ids.extend(encoding["input_ids"])
        attention_mask.extend(encoding["attention_mask"])

    # create dataset
    dataset = Dataset.from_dict({"input_ids": input_ids,
                                  "attention_mask": attention_mask,
                                  "labels": df["label"].tolist()})
    return dataset, df