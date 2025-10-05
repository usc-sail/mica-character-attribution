"""Extract character segments from script"""
import data_utils

from absl import app
from absl import flags
import collections
import numpy as np
import os
import pandas as pd
import re
import tqdm
from typing import List, Tuple

flags.DEFINE_bool("anonymize", default=False, help="anonymize character names")
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

def find_character_spans_and_create_paragraph_from_segments(
        segment_texts: List[str],
        segment_character_spans_list: List[List[List[int]]],
        segment_delimiter = " ") -> Tuple[str, List[List[int]]]:
    """
    Concatenate segment texts into a single paragraph and remap the character spans in the segments to the paragraph
    """

    # concatenate the segment texts and remap character spans
    paragraph = ""
    paragraph_character_spans = []
    offset = 0
    for segment_text, segment_character_spans in zip(segment_texts, segment_character_spans_list):
        if segment_character_spans:
            for start, end in segment_character_spans:
                paragraph_character_spans.append([start + offset, end + offset])
        paragraph += segment_text + segment_delimiter
        offset += len(segment_text) + len(segment_delimiter)

    # remove the extra trailing segment_delimiter
    if segment_texts:
        paragraph = paragraph[: -len(segment_delimiter)]

    # replace contiguous whitespaces between adjacent character spans with a single space
    for i in range(len(paragraph_character_spans)):

        # find the span of text that needs to be cleaned
        # for i > 0, this will be the text between the end of the i-1 th span and beginning of the i-th span
        # for i == 0, this will be the text before the beginning of the 0th span
        if i == 0:
            start = 0
            end = paragraph_character_spans[i][0]
        else:
            start = paragraph_character_spans[i - 1][1]
            end = paragraph_character_spans[i][0]

        # clean the text and find the difference in length before and after cleaning
        # the difference will be used to change the ith and following spans
        initial_length = len(paragraph[start: end])
        cleaned_text = re.sub("\s+", " ", paragraph[start: end])
        final_length = len(cleaned_text)
        paragraph = paragraph[:start] + cleaned_text + paragraph[end:]
        offset = final_length - initial_length

        # change the ith and following spans
        for j in range(i, len(paragraph_character_spans)):
            paragraph_character_spans[j] = [paragraph_character_spans[j][0] + offset,
                                            paragraph_character_spans[j][1] + offset]

    # check if the non-whitespace characters remain the same
    segment_texts_non_whitespace_text = re.sub("\s+", "", "".join(segment_texts))
    paragraph_non_whitespace_text = re.sub("\s+", "", paragraph)
    assert segment_texts_non_whitespace_text == paragraph_non_whitespace_text, (
        "non whitespace text differ between segments and paragraphs --> something has gone wrong!")

    # check if the segment_spans and paragraph_spans map to the same text
    i = 0
    for segment_text, segment_character_spans in zip(segment_texts, segment_character_spans_list):
        for start, end in segment_character_spans:
            segment_span_text = segment_text[start: end]
            paragraph_span_start, paragraph_span_end = paragraph_character_spans[i]
            paragraph_span_text = paragraph[paragraph_span_start: paragraph_span_end]
            assert segment_span_text == paragraph_span_text, "segment and paragraph span texts don't match!"
            i += 1

    return paragraph, paragraph_character_spans

def extract_segments(_):
    """Extract segments from movie script where character is mentioned"""
    # get file paths
    movie_scripts_dir = os.path.join(data_utils.DATADIR, "CHATTER/movie-scripts")
    map_file = os.path.join(data_utils.DATADIR, "CHATTER/character-movie-map.csv")
    output_dir = os.path.join(data_utils.DATADIR, "CHATTER", "new-anonymized-segments" if FLAGS.anonymize else "new-segments")
    os.makedirs(output_dir, exist_ok=True)

    # read data
    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    imdbid_to_segments_df = {}
    imdbid_to_mentions_df = {}
    for imdbid in tqdm.tqdm(map_df["imdb-id"].unique(), desc="read movie scripts"):
        segments_file = os.path.join(movie_scripts_dir, imdbid, "segments.csv")
        mentions_file = os.path.join(movie_scripts_dir, imdbid, "mentions.csv")
        segments_df = pd.read_csv(segments_file, index_col=None)
        mentions_df = pd.read_csv(mentions_file, index_col=None)
        imdbid_to_segments_df[imdbid] = segments_df
        imdbid_to_mentions_df[imdbid] = mentions_df

    # anonymize segments
    if FLAGS.anonymize:
        name_and_imdbid_to_characterid = {}
        for _, row in map_df.iterrows():
            name_and_imdbid_to_characterid[(row["name"], row["imdb-id"])] = row["character"]
        n = map_df["character"].str[1:].astype(int).max() + 1
        for imdbid in tqdm.tqdm(imdbid_to_segments_df.keys(), desc="anonymizing"):
            segments_df = imdbid_to_segments_df[imdbid]
            mentions_df = imdbid_to_mentions_df[imdbid]
            segments_df = segments_df.merge(mentions_df, how="left", on="segment-id")
            segment_rows = []
            mention_rows = []
            for segment_id, segment_df in segments_df.groupby("segment-id", sort=False):
                offset = 0
                if pd.notna(segment_df["imdb-character"].values[0]):
                    segment_df = segment_df.sort_values("start", ascending=True)
                    segment_text = segment_df["segment-text"].values[0]
                    for _, row in segment_df.iterrows():
                        name, start, end = row["imdb-character"], int(row["start"]), int(row["end"])
                        start += offset
                        end += offset
                        if (name, imdbid) in name_and_imdbid_to_characterid:
                            characterid = name_and_imdbid_to_characterid[(name, imdbid)]
                        else:
                            characterid = "C" + str(n).zfill(4)
                            name_and_imdbid_to_characterid[(row["imdb-character"], imdbid)] = characterid
                            n += 1
                        charactername = "CHARACTER" + characterid[1:]
                        segment_text = segment_text[:start] + charactername + segment_text[end:]
                        end = start + len(charactername)
                        mention_rows.append([segment_id, name, start, end])
                        offset += len(charactername) - (end - start)
                    segment_rows.append([segment_id, segment_text])
                else:
                    segment_rows.append([segment_id, segment_df["segment-text"].values[0]])
            segments_df = pd.DataFrame(segment_rows, columns=["segment-id", "segment-text"])
            mentions_df = pd.DataFrame(mention_rows, columns=["segment-id", "imdb-character", "start", "end"])
            imdbid_to_segments_df[imdbid] = segments_df
            imdbid_to_mentions_df[imdbid] = mentions_df

    # process data
    paragraph_sizes = []
    segment_sizes = []
    character_sizes = []
    n_paragraphs_per_segment = []
    n_segments_per_character = []
    n_paragraphs_per_character = []
    for characterid, character_df in tqdm.tqdm(map_df.groupby("character"), desc="find character segments", 
                                               total=len(map_df["character"].unique()), unit="character"):
        n_character_segments = 0
        n_character_paragraphs = 0
        character_size = 0
        for imdbid, name in character_df[["imdb-id", "name"]].itertuples(index=False, name=None):
            segments_df, mentions_df = imdbid_to_segments_df[imdbid], imdbid_to_mentions_df[imdbid]
            character_segmentids_to_spans = collections.defaultdict(list)
            for _, row in mentions_df.iterrows():
                if row["imdb-character"] == name:
                    character_segmentids_to_spans[row["segment-id"]].append([row["start"], row["end"]])
            if character_segmentids_to_spans:
                n_character_segments += 1
                character_segment_paragraphs = []
                character_paragraph_spans = []
                segmentids = segments_df["segment-id"].tolist()
                segment_texts = segments_df["segment-text"].tolist()
                i = 0
                while i < len(segmentids):
                    if segmentids[i] in character_segmentids_to_spans:
                        j = i + 1
                        while j < len(segmentids) and segmentids[j] in character_segmentids_to_spans:
                            j = j + 1
                        character_segment_texts = segment_texts[i: j]
                        character_segment_spans_list = [character_segmentids_to_spans[segmentid]
                                                        for segmentid in segmentids[i: j]]
                        paragraph, paragraph_spans = find_character_spans_and_create_paragraph_from_segments(
                            character_segment_texts, character_segment_spans_list)
                        paragraph_sizes.append(len(paragraph.split()))
                        n_character_paragraphs += 1
                        character_segment_paragraphs.append(paragraph)
                        character_paragraph_spans.append(paragraph_spans)
                        i = j
                    else:
                        i = i + 1
                n_paragraphs_per_segment.append(len(segment_texts))
                character_segments_file = os.path.join(output_dir, f"{characterid}-{imdbid}.txt")
                character_segments_spans_file = os.path.join(output_dir, f"{characterid}-{imdbid}-spans.txt")
                character_segments_text = "\n".join(character_segment_paragraphs)
                character_segments_spans = []
                offset = 0
                for i, paragraph_spans in enumerate(character_paragraph_spans):
                    for start, end in paragraph_spans:
                        character_segments_spans.append([start + offset, end + offset])
                    offset += len(character_segment_paragraphs[i]) + 1
                segment_size = len(character_segments_text.split())
                segment_sizes.append(segment_size)
                character_size += segment_size
                with open(character_segments_file, "w") as fw:
                    fw.write(character_segments_text)
                with open(character_segments_spans_file, "w") as fw:
                    for start, end in character_segments_spans:
                        fw.write(f"{start} {end}\n")
        character_sizes.append(character_size)
        n_segments_per_character.append(n_character_segments)
        n_paragraphs_per_character.append(n_character_paragraphs)

    # find statistics
    stats_file = os.path.join(output_dir, "stats.txt")
    with open(stats_file, "w") as fw:
        write_statistics(paragraph_sizes, "words per paragraph", fw)
        write_statistics(segment_sizes, "words per segment", fw)
        write_statistics(character_sizes, "words per character", fw)
        write_statistics(n_paragraphs_per_segment, "paragraphs per segment", fw)
        write_statistics(n_segments_per_character, "segments per character", fw)
        write_statistics(n_paragraphs_per_character, "paragraphs per character", fw)

if __name__ == '__main__':
    app.run(extract_segments)