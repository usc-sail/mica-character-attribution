"""Find character mentions and character utterances in scripts. The output is a character mentions and a character
utterances csv file. These files are saved inside the corresponding movie subdirectory of the movie-scripts directory.

The character mentions csv file contains the following fields: "imdb-character", "segment-id", "start" and
"end". The "start" and "end" define the span of the character mention in the script segment identified by "segment-id".

The character utterances csv file contains the following fields: "imdb-character", "segment-id", "start" and "end".
The "segment-id" identifies the utterance segment in the script.
"""
import collections
import json
import nameparser
import numpy as np
import os
import pandas as pd
import re
import spacy
import spacy.tokens
import tqdm

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("data_file", default="20-preprocessing/train-dev-test-splits-with-negatives.csv", help="data file")
flags.DEFINE_string("scripts_dir", default="movie-scripts", help="scripts directory")
flags.DEFINE_integer("k", default=10, help="top k cast to consider for each movie")

def get_aliases_from_name(name):
    """Find aliases of name"""
    hname = nameparser.HumanName(name)
    hnames = [hname]
    if hname.first and hname.last:
        hnames.append(nameparser.HumanName(f"{hname.first} {hname.last}"))
    if hname.title and hname.first:
        hnames.append(nameparser.HumanName(f"{hname.title} {hname.first}"))
    if hname.title and hname.last:
        hnames.append(nameparser.HumanName(f"{hname.title} {hname.last}"))
    if hname.first:
        hnames.append(nameparser.HumanName(f"{hname.first}"))
    if hname.last:
        hnames.append(nameparser.HumanName(f"{hname.last}"))
    aliases = set()
    for hname in hnames:
        hname.capitalize(force=True)
        aliases.update([str(hname), str(hname).upper()])
    aliases.update([name, name.upper()])
    return sorted(aliases)

def get_topk_characters(imdb_data, k):
    """Get top k characters from the IMDB cast list"""
    characters = set()
    if "cast" in imdb_data:
        for character in imdb_data["cast"][:k]:
            if "character" in character:
                name = character["character"]
                if name is not None:
                    characters.add(name.strip())
    return characters

def normalize_name(name):
    """Remove parantheses, match pattern, and consolidate whitespace"""
    name = re.sub(r"\([^\)]\)", "", name).strip()
    if re.match(r"[a-zA-Z0-9\s\.\-\']+$", name) is not None and len(re.findall(r"[\.\-\']", name)) <= 2:
        return name

def match_script_name_to_imdb_name(script_name, imdb_name):
    """Match script name to IMDB name"""
    tokens = set(script_name.lower().split())
    parsed_imdb_name = nameparser.HumanName(imdb_name)
    imdb_name_tokens = [parsed_imdb_name.title.lower(), parsed_imdb_name.first.lower(), parsed_imdb_name.last.lower(),
                        parsed_imdb_name.middle.lower()]
    imdb_name_tokens = set(filter(lambda x: len(x) > 0, imdb_name_tokens))
    return len(tokens.intersection(imdb_name_tokens))

def match_script_names_to_imdb_names(script_names, imdb_names):
    """Try to match a script name to atmost one imdb name"""
    n_matches = np.zeros(len(imdb_names), dtype=int)
    for i, script_name in enumerate(script_names):
        n_matches.fill(0)
        for j, imdb_name in enumerate(imdb_names):
            n_matches[j] = match_script_name_to_imdb_name(script_name, imdb_name)
        j = np.argmax(n_matches)
        n_maxes = (n_matches == n_matches[j]).sum()
        if n_maxes == 1 and n_matches[j] > 0:
            yield i, j

def find_character_mentions(_):
    data_dir = FLAGS.data_dir
    data_file = os.path.join(data_dir, FLAGS.data_file)
    scripts_dir = os.path.join(data_dir, FLAGS.scripts_dir)
    k = FLAGS.k

    nlp = spacy.blank("en")
    data_df = pd.read_csv(data_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
    n, n1, n2 = 0, 0, 0

    groups = data_df.groupby("imdb-id")
    for imdb_id, imdb_df in tqdm.tqdm(groups, total=groups.ngroups):
        matches_file = os.path.join(scripts_dir, imdb_id, "name-matches.txt")
        mentions_file = os.path.join(scripts_dir, imdb_id, "mentions.csv")
        utterances_file = os.path.join(scripts_dir, imdb_id, "utterances.csv")
        mention_rows = []
        utterance_rows = []

        with open(matches_file, "w") as fw:

            data_imdb_names = sorted(imdb_df["imdb-character"].str.strip().unique())
            segments_file = os.path.join(scripts_dir, imdb_id, "segments.csv")
            spacy_segments_file = os.path.join(scripts_dir, imdb_id, "spacy-segments.bytes")
            imdb_file = os.path.join(scripts_dir, imdb_id, "imdb.json")

            segments_df = pd.read_csv(segments_file, index_col=None)
            with open(spacy_segments_file, "rb") as fr:
                doc_bin = spacy.tokens.DocBin().from_bytes(fr.read())
            docs = list(doc_bin.get_docs(nlp.vocab))
            imdb_data = json.load(open(imdb_file))

            script_names = set()
            for doc in docs:
                for ent in doc.ents:
                    if ent.label == "PERSON":
                        script_names.add(ent.text)

            script_names.update(segments_df["segment-speaker"].dropna())
            script_names = list(filter(lambda x: x is not None, map(normalize_name, script_names)))
            script_name_counts = collections.Counter(script_names)
            script_names = sorted([name for name, count in script_name_counts.most_common(2*k) if count > 1])

            imdb_names = sorted(get_topk_characters(imdb_data, k))
            imdb_name_aliases = {name:get_aliases_from_name(name) for name in imdb_names}
            script_names = script_names + [name_alias for name_aliases in imdb_name_aliases.values()
                                            for name_alias in name_aliases]
            script_names = sorted(set(script_names))

            imdb_name_to_matched_names = collections.defaultdict(list)
            for i, j in match_script_names_to_imdb_names(script_names, imdb_names):
                imdb_name_to_matched_names[imdb_names[j]].append(script_names[i])

            fw.write(imdb_id + "\n")
            fw.write("imdb-names\n\t" + "\n\t".join(imdb_names) + "\n")
            fw.write("script-names\n\t" + "\n\t".join(script_names) + "\n\n")
            fw.write("matches\n")

            for imdb_name in data_imdb_names:
                aliases = imdb_name_aliases[imdb_name]
                matched_names = imdb_name_to_matched_names[imdb_name]
                n_mentions, n_utterances = 0, 0
                if matched_names:
                    pattern = r"(^|\W)(" + r"|".join([r"(" + re.escape(matched_name) + r")" 
                                                        for matched_name in matched_names]) + r")(\W|$)"
                    for _, row in segments_df.iterrows():
                        segment_id, segment_speaker, segment_text = (row["segment-id"], row["segment-speaker"],
                                                                        row["segment-text"])
                        for match in re.finditer(pattern, segment_text):
                            i, j = match.span(2)
                            mention_rows.append([imdb_name, segment_id, i, j, segment_text[i: j]])
                            n_mentions += 1
                        if pd.notna(segment_speaker):
                            match = re.search(pattern, segment_speaker)
                            if match is not None:
                                i, j = match.span(2)
                                utterance_rows.append([imdb_name, segment_id, i, j, segment_speaker[i: j]])
                                n_utterances += 1

                parsed_imdb_name = nameparser.HumanName(imdb_name)
                title, first, middle, last = (parsed_imdb_name.title, parsed_imdb_name.first, parsed_imdb_name.middle,
                                                parsed_imdb_name.last)
                fw.write(f"\t{imdb_name} {[title, first, middle, last]} ({n_mentions} mentions {n_utterances} utter)\n")
                for alias in aliases:
                    if alias in matched_names:
                        fw.write(f"\t\t{alias} (matched)\n")
                    else:
                        fw.write(f"\t\t{alias}\n")

        mention_df = pd.DataFrame(mention_rows, columns=["imdb-character", "segment-id", "start", "end", "text"])
        utterance_df = pd.DataFrame(utterance_rows, columns=["imdb-character", "segment-id", "start", "end", "text"])

        n += len(imdb_df["imdb-character"].unique())
        n1 += len(mention_df["imdb-character"].unique())
        n2 += len(utterance_df["imdb-character"].unique())

        mention_df.to_csv(mentions_file, index=False)
        utterance_df.to_csv(utterances_file, index=False)

    print(f"{n} imdb-characters, {n1} has mentions, {n2} has utterances")

if __name__ == '__main__':
    app.run(find_character_mentions)