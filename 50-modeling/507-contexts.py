"""Extract trope-relevant paragraphs from character segments"""
import data_utils

from absl import app
from absl import flags
import collections
import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
from sentence_transformers import SentenceTransformer
import torch
import tqdm

flags.DEFINE_integer("min_words_per_paragraph", default=25,
                     help=("minimum number of words in concatenated paragraph (set to 1 or less if you do not want to "
                           "concatenate)"))
flags.DEFINE_integer("max_words_per_context", default=2400, help="maximum number of words per context", 
                     upper_bound=2500)
flags.DEFINE_enum("strategy", default="trope", enum_values=["random", "first", "last", "trope"],
                  help="sampling strategy for paragraphs")
flags.DEFINE_string("model", default="sentence-transformers/all-mpnet-base-v2", help="sentence encoder model")
flags.DEFINE_float("min_pos_similarity", default=0.05, help="minimum positive similarity", lower_bound=0)
flags.DEFINE_float("max_neg_similarity", default=-0.05, help="maximum negative similarity", upper_bound=0)
flags.DEFINE_bool("anonymize", default=False, help="use anonymized segments")
flags.DEFINE_integer("seed", default=2050, help="seed for randomizer")
FLAGS = flags.FLAGS

def write_statistics(arr, name, fw):
    mean = np.mean(arr)
    _max = max(arr)
    _min = min(arr)
    median = np.median(arr)
    std = np.std(arr)
    fw.write(f"{name}:\n")
    fw.write(f"mean = {mean:.2f}\n")
    fw.write(f"std = {std:.2f}\n")
    fw.write(f"min = {_min}\n")
    fw.write(f"max = {_max}\n")
    fw.write(f"median = {median:.2f}\n")
    for f in [0.8, 0.9, 0.95, 0.98, 0.99]:
        fw.write(f"{f} %tile = {np.quantile(arr, f):.2f}\n")
    fw.write("\n")

def extract_contexts(_):
    """Extracts contexts from character segments"""
    # get filepaths
    label_file = os.path.join(data_utils.DATADIR, "CHATTER/chatter.csv")
    map_file = os.path.join(data_utils.DATADIR, "CHATTER/character-movie-map.csv")
    tropes_file = os.path.join(data_utils.DATADIR, "CHATTER/tropes.csv")
    segments_dir = os.path.join(data_utils.DATADIR,
                                "50-modeling",
                                "anonymized-segments" if FLAGS.anonymize else "segments")
    random.seed(FLAGS.seed)

    # read data
    label_df = pd.read_csv(label_file, index_col=None)
    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    characterid_and_imdbid_to_paragraphs = {}
    characterid_and_imdbid_to_name = {}
    characterid_to_imdbids = collections.defaultdict(list)
    tropes = tropes_df.trope.tolist()
    definitions = tropes_df["summary"].tolist()
    trope_to_ix = {trope: i for i, trope in enumerate(tropes)}
    n_words_per_paragraph = []
    for filename in tqdm.tqdm(os.listdir(segments_dir), desc="reading segments"):
        match = re.match(r"(C\d+)-(\d+)\.txt", filename)
        if match is not None:
            characterid = match.group(1)
            imdbid = match.group(2)
            segment_file = os.path.join(segments_dir, filename)
            with open(segment_file) as fr:
                paragraphs = fr.read().strip().split("\n")
            nwords = [len(paragraph.split()) for paragraph in paragraphs]
            concatenated_paragraphs = []
            i = 0
            while i < len(paragraphs):
                total_nwords = nwords[i]
                j = i + 1
                while j < len(paragraphs) and total_nwords + nwords[j] <= FLAGS.min_words_per_paragraph:
                    total_nwords += nwords[j]
                    j += 1
                concatenated_paragraph = "\n".join(paragraphs[i: j])
                concatenated_paragraphs.append(concatenated_paragraph)
                n_words_per_paragraph.append(len(concatenated_paragraph.split()))
                i = j
            characterid_and_imdbid_to_paragraphs[(characterid, imdbid)] = concatenated_paragraphs
            characterid_to_imdbids[characterid].append(imdbid)
    for _, row in map_df.iterrows():
        characterid_and_imdbid_to_name[(row.character, row["imdb-id"])] = row["name"]

    # instantiate model and encode tropes
    if FLAGS.strategy == "trope":
        print("instantiating model")
        encoder = SentenceTransformer(FLAGS.model, device="cuda:0", similarity_fn_name="cosine")
        print("encoding tropes")
        definition_tensors = encoder.encode(definitions, convert_to_tensor=True, device="cuda:0", batch_size=256,
                                            show_progress_bar=True)

    # find contexts
    contexts_dir = "anonymized-contexts" if FLAGS.anonymize else "contexts"
    if FLAGS.strategy == "trope":
        modelname = FLAGS.model.split("/")[-1]
        lb = abs(FLAGS.max_neg_similarity)
        strategy = f"{modelname}-{lb}NEG-{FLAGS.min_pos_similarity}POS"
        histfilename = f"{FLAGS.min_words_per_paragraph}P-{FLAGS.max_words_per_context}C-{modelname}"
        histfile = os.path.join(data_utils.DATADIR, f"50-modeling/{contexts_dir}/{histfilename}-similarities.png")
    else:
        strategy = FLAGS.strategy
    filename = f"{FLAGS.min_words_per_paragraph}P-{FLAGS.max_words_per_context}C-{strategy}"
    contexts_file = os.path.join(data_utils.DATADIR, f"50-modeling/{contexts_dir}/{filename}.jsonl")
    stats_file = os.path.join(data_utils.DATADIR, f"50-modeling/{contexts_dir}/{filename}-stats.txt")
    label_df = label_df[label_df["partition"].notna()]
    n_words_per_context = []
    n_paras_per_context = []
    n_excluded_paras_per_context = []
    similarities = []
    objs = []
    for characterid, character_df in tqdm.tqdm(label_df.groupby("character"),
                                                total=label_df.character.unique().size,
                                                desc="extract contexts", unit="character"):
        if FLAGS.strategy == "trope":
            character_tropes = character_df.trope.tolist()
            partitions = character_df.partition.tolist()
            labels = []
            for _, row in character_df.iterrows():
                label = row.label if row.partition == "test" else row["tvtrope-label"]
                labels.append(int(label))
            character_tropes_ix = [trope_to_ix[trope] for trope in character_tropes]
            trope_definitions = [definitions[i] for i in character_tropes_ix]
            character_trope_tensors = definition_tensors[character_tropes_ix]
        for imdbid in characterid_to_imdbids[characterid]:
            name = characterid_and_imdbid_to_name[(characterid, imdbid)]
            paragraphs = characterid_and_imdbid_to_paragraphs[(characterid, imdbid)]
            nwords = [len(paragraph.split()) for paragraph in paragraphs]
            paragraphs_ix = list(range(len(paragraphs)))
            if FLAGS.strategy != "trope":
                if FLAGS.strategy == "random":
                    random.shuffle(paragraphs_ix)
                elif FLAGS.strategy == "last":
                    paragraphs_ix = list(reversed(paragraphs_ix))
                i = 0
                total_nwords = 0
                while i < len(nwords) and total_nwords + nwords[paragraphs_ix[i]] <= FLAGS.max_words_per_context:
                    total_nwords += nwords[paragraphs_ix[i]]
                    i += 1
                context_paragraphs = [paragraphs[j] for j in sorted(paragraphs_ix[:i])]
                context = "\n".join(context_paragraphs)
                n_words_per_context.append(len(context.split()))
                n_paras_per_context.append(i)
                n_excluded_paras_per_context.append(len(nwords) - i)
                obj = {"character": characterid, "imdbid": imdbid, "name": name, "text": context}
                objs.append(obj)
            else:
                paragraph_tensors = encoder.encode(paragraphs, convert_to_tensor=True, device="cuda:0",
                                                    show_progress_bar=False)
                similarity = encoder.similarity(character_trope_tensors, paragraph_tensors)
                similarities.extend(similarity.flatten().tolist())
                similarity[(similarity > FLAGS.max_neg_similarity)
                            & (similarity < FLAGS.min_pos_similarity)] = torch.nan
                similarity = torch.abs(similarity)
                similarity = torch.nan_to_num(similarity, -torch.inf)
                argsort = torch.argsort(similarity, descending=True)
                isinf = torch.isinf(similarity)
                for i, (trope, definition, label, partition) in enumerate(zip(character_tropes, trope_definitions,
                                                                              labels, partitions)):
                    args = argsort[i]
                    j = 0
                    total_nwords = 0
                    while j < len(paragraphs) and not isinf[i, args[j]] and (
                        total_nwords + nwords[args[j]] <= FLAGS.max_words_per_context):
                        total_nwords += nwords[args[j]]
                        j += 1
                    context_paragraphs = [paragraphs[k] for k in sorted(args[:j])]
                    context = "\n".join(context_paragraphs)
                    n_words_per_context.append(len(context.split()))
                    n_paras_per_context.append(j)
                    n_excluded_paras_per_context.append(len(nwords) - j)
                    obj = {"character": characterid, "imdbid": imdbid, "name": name, "trope": trope,
                           "definition": definition, "partition": partition, "label": label, "text": context}
                    objs.append(obj)

    # save output
    with jsonlines.open(contexts_file, "w") as writer:
        writer.write_all(objs)
    with open(stats_file, "w") as fw:
        write_statistics(n_words_per_paragraph, "words/paragraph", fw)
        write_statistics(n_words_per_context, "words/context", fw)
        write_statistics(n_paras_per_context, "paras/context", fw)
        write_statistics(n_excluded_paras_per_context, "excluded-paras/context", fw)
    if FLAGS.strategy == "trope":
        plt.hist(similarities, bins=100, density=True)
        plt.xlabel("similarity")
        plt.title("histogram")
        plt.savefig(histfile)

if __name__ == '__main__':
    app.run(extract_contexts)