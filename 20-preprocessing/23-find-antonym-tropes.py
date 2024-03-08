"""Find the antonym tropes of each trope from their definition.
Process the definition to find tokens, sentence boundaries, and linked tropes.
"""
import os
import re
import json
import tqdm
import spacy
import pandas as pd
from bs4 import BeautifulSoup as bs

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("tropes", default="10-crawling/tropes.csv", help="csv file of trope, definition, and linked-tropes")
flags.DEFINE_string("docs", default="20-preprocessing/tropes-docs.json", 
                    help="json file of spacy-tokenized tokens of trope definitions")
flags.DEFINE_string("antonyms", default="20-preprocessing/tropes-with-antonyms.csv", 
                    help="csv file of trope, definition, linked-tropes, and antonym-tropes")
flags.DEFINE_integer("window", default=5, help="window around trope to find negation words")
flags.DEFINE_integer("gpu_id", default=0, help="gpu id to use for spacy preprocessing")

def find_antonym_tropes(_):
    # list of antonym words to search in the context of tropes in the definition
    antonym_words = set(["contrast", "opposite", "not", "never", "nor", "opposite", "inverse", "conflict", "conflicts", 
                         "unlike", "different", "inversion", "counterpart", "difference", "counterparts", "differences",
                         "opposites"])

    # files and folders
    data_dir = FLAGS.data_dir
    tropes_file = os.path.join(data_dir, FLAGS.tropes)
    docs_file = os.path.join(data_dir, FLAGS.docs)
    antonyms_file = os.path.join(data_dir, FLAGS.antonyms)
    window = FLAGS.window
    gpu_id = FLAGS.gpu_id

    # read tropes dataframe
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    tropes_df = tropes_df[tropes_df["definition"].notna()]
    tropes_df = tropes_df.drop_duplicates("trope")
    tropes = tropes_df["trope"].tolist()
    print(f"{len(tropes_df)} tropes")

    # load spacy model
    print("loading spacy ...")
    spacy.require_gpu(gpu_id)
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
    nlp.add_pipe("sentencizer")

    # parse trope definitions
    definitions = []
    soups = []
    for definition in tqdm.tqdm(tropes_df["definition"], desc="parsing html"):
        definition = re.sub(r"\s+", " ", definition).strip()
        soup = bs(definition, features="html.parser")
        definitions.append(soup.text)
        soups.append(soup)

    # process definitions using spacy
    nlp_docs = []
    for nlp_doc in tqdm.tqdm(nlp.pipe(definitions, batch_size=512), total=len(definitions), desc="tokenizing"):
        nlp_docs.append(nlp_doc)

    # tokenize and find antonyms
    docs = {}
    antonyms_arr = []
    for trope, definition, soup, nlp_doc in tqdm.tqdm(zip(tropes, definitions, soups, nlp_docs), 
                                                      total=len(definitions), desc="finding antonym tropes"):
        # find offset to token map
        offset = 0
        offset2trope = {}
        for child in soup.children:
            size = len(child.text)
            if child.name == "trope":
                offset2trope[offset] = (size, child["value"].split("/")[-1])
            offset += size

        # find tokens, sent index, and type arrays
        tokens = []
        sents = []
        types = []
        i = 0
        while i < len(nlp_doc):
            offset = nlp_doc[i].idx
            if offset in offset2trope:
                size, value = offset2trope[offset]
                j = i
                span_size = 0
                while j < len(nlp_doc) and span_size < size:
                    span_size = nlp_doc[j].idx + len(nlp_doc[j]) - offset
                    types.append(value)
                    j = j + 1
                i = j
            else:
                types.append("token")
                i = i + 1

        for i, sent in enumerate(nlp_doc.sents):
            for token in sent:
                tokens.append(token.text)
                sents.append(i)

        # find antonym tropes
        antonyms = set()
        antonym_arr = []
        i = 0
        while i < len(tokens):
            if types[i] != "token":
                j = i - 1
                k = i + 1
                while k < len(tokens) and types[k] == types[i]:
                    k += 1
                l = k
                left_tokens, right_tokens = set(), set()
                while j >= 0 and sents[j] == sents[i] and len(left_tokens) < window:
                    if types[j] == "token" and tokens[j] not in ["and", ","]:
                        left_tokens.add(tokens[j].lower())
                    j = j - 1
                while l < len(tokens) and sents[l] == sents[i] and len(right_tokens) < window:
                    if types[l] == "token" and tokens[l] not in ["and", ","]:
                        right_tokens.add(tokens[l].lower())
                    l = l + 1
                if left_tokens.union(right_tokens).intersection(antonym_words):
                    antonyms.add(types[i])
                    antonym_arr.extend([1 for _ in range(k - i)])
                else:
                    antonym_arr.extend([0 for _ in range(k - i)])
                i = k
            else:
                i = i + 1
                antonym_arr.append(0)
        antonyms = sorted(antonyms)

        docs[trope] = {"text": definition, "token": tokens, "sent": sents, "trope": types, "antonym-trope": antonym_arr}
        antonyms_arr.append(antonyms)

    # write antonyms
    antonyms_set = set([antonym for antonyms in antonyms_arr for antonym in antonyms])
    percentage = 100 * len(antonyms_set.intersection(tropes)) / len(antonyms_set)
    print(f"{len(antonyms_set)} antonyms tropes, {percentage:.1f}% already crawled")
    tropes_df["antonym-tropes"] = [";".join(antonyms) for antonyms in antonyms_arr]
    tropes_df.to_csv(antonyms_file, index=False)

    # write trope docs
    with open(docs_file, "w") as fw:
        json.dump(docs, fw, indent=2)

if __name__ == '__main__':
    app.run(find_antonym_tropes)