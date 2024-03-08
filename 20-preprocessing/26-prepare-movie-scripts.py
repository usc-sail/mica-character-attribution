"""Copy movie scripts to data directory, separate the segments, and spacify each segment"""
import os
import pandas as pd
import re
import shutil
import spacy
from spacy import tokens
import tqdm
import unidecode

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("scripts_dir", default=None, help="scripts directory", required=True)
flags.DEFINE_string("output_scripts_dir", default="movie-scripts", help="output directory to save movie scripts")
flags.DEFINE_string("data_file", default="20-preprocessing/train-dev-test-splits-with-negatives.csv",
                    help="csv file containing imdb-ids of the movie scripts dataset")

def clean_text(text):
    text = unidecode.unidecode(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_movie_scripts(_):
    data_dir = FLAGS.data_dir
    scripts_dir = FLAGS.scripts_dir
    output_dir = os.path.join(data_dir, FLAGS.output_scripts_dir)
    data_file = os.path.join(data_dir, FLAGS.data_file)
    spacy.require_gpu()
    nlp = spacy.load("en_core_web_sm")

    data_df = pd.read_csv(data_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
    imdb_ids = data_df["imdb-id"].unique()
    pattern = r"(S+)|(N+)|((C+)([EDX]+))"

    for imdb_id in tqdm.tqdm(imdb_ids):
        movie_output_dir = os.path.join(output_dir, imdb_id)
        os.makedirs(movie_output_dir, exist_ok=True)
        src_script_file = os.path.join(scripts_dir, imdb_id, "script.txt")
        src_parse_file = os.path.join(scripts_dir, imdb_id, "trfr-parse.txt")
        src_imdb_file = os.path.join(scripts_dir, imdb_id, "imdb.json")
        dst_script_file = os.path.join(movie_output_dir, "script.txt")
        dst_parse_file = os.path.join(movie_output_dir, "parse.txt")
        dst_imdb_file = os.path.join(movie_output_dir, "imdb.json")
        shutil.copyfile(src_script_file, dst_script_file)
        shutil.copyfile(src_parse_file, dst_parse_file)
        shutil.copyfile(src_imdb_file, dst_imdb_file)
        script_lines = open(dst_script_file, encoding="utf-8").read().strip().split("\n")
        parse = open(dst_parse_file).read().strip().replace("\n", "")
        parse = re.sub(r"[OTM]", "X", parse)
        assert len(script_lines) == len(parse)

        segment_rows, segment_texts = [], []
        last_match_end, n_segments = 0, 0
        for match in tqdm.tqdm(re.finditer(pattern, parse), desc="segmenting", leave=False):
            i, j = match.span(0)
            if i > last_match_end:
                other_text = " ".join(script_lines[last_match_end: i])
                other_text = clean_text(other_text)
                if other_text:
                    segment_id = f"{imdb_id}-other-{n_segments}"
                    segment_rows.append([segment_id, "other", other_text, ""])
                    segment_texts.append(other_text)
                    n_segments += 1
            text = " ".join(script_lines[i: j])
            text = clean_text(text)
            if text:
                segment_texts.append(text)
                segment_type, segment_speaker = "", ""
                if match.group(1):
                    segment_id = f"{imdb_id}-slugline-{n_segments}"
                    segment_type = "slugline"
                elif match.group(2):
                    segment_id = f"{imdb_id}-desc-{n_segments}"
                    segment_type = "desc"
                else:
                    k, l = match.span(4)
                    segment_id = f"{imdb_id}-utter-{n_segments}"
                    segment_type = "utter"
                    segment_speaker = " ".join(script_lines[k: l])
                    segment_speaker = clean_text(segment_speaker)
                n_segments += 1
                segment_rows.append([segment_id, segment_type, text, segment_speaker])
            last_match_end = j

        docbin = tokens.DocBin()
        for doc in tqdm.tqdm(nlp.pipe(segment_texts, batch_size=1024), total=len(segment_texts), desc="spacify",
                             leave=False):
            docbin.add(doc)

        segments_file = os.path.join(movie_output_dir, "segments.csv")
        docs_file = os.path.join(movie_output_dir, "spacy-segments.bytes")
        segments_df = pd.DataFrame(segment_rows, columns=["segment-id", "segment-type", "segment-text",
                                                          "segment-speaker"])
        segments_df.to_csv(segments_file, index=False)
        with open(docs_file, "wb") as fw:
            fw.write(docbin.to_bytes())

if __name__ == '__main__':
    app.run(prepare_movie_scripts)