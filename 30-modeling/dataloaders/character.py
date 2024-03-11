"""Tensorize mentions and utterances of a character"""
from absl import logging
import numpy as np
import pandas as pd
import torch

def create_tensors(tokenizer, imdb_character, segments_df, mentions_df, utterances_df):
    """Tensorize the data for training and testing. The data will be of the form [CLS] t1 t2 ... tn [SEP] where t1, t2,
    ..., tn are tokens.
    Input
    =====
        tokenizer = huggingface model tokenizer
        imdb_character = imdb character name
        segments_df = dataframe of segments
        mentions_df = dataframe of mentions
        utterances_df = dataframe of utterances
    Output
    ======
        character_token_ids = long tensor [n_blocks, model_max_length]
        mentions_mask = float tensor [n_mentions, n_blocks x model_max_length]
        utterances_mask = float tensor [n_utterances, n_blocks x model_max_length]
    """
    max_n_tokens_single_sentence = tokenizer.max_len_single_sentence

    # join segments dataframe with the mention and utterance spans dataframes
    segments_df["n-tokens"] = segments_df["segment-text"].apply(
        lambda text: len(tokenizer(text, add_special_tokens=False).input_ids))
    segments_df["segment-order"] = segments_df["segment-id"].str.split("-").str[-1].astype(int)
    mentions_df["span-type"] = "mention"
    utterances_df["span-type"] = "utterance"
    utterances_df["start"] = 0
    spans_df = pd.concat([mentions_df, utterances_df], axis=0)
    spans_df = spans_df[spans_df["imdb-character"] == imdb_character]
    n_mentions = (spans_df["span-type"] == "mention").sum()
    n_utterances = (spans_df["span-type"] == "utterance").sum()
    segments_df = segments_df.merge(spans_df, how="inner", on="segment-id")
    segments_df.loc[segments_df["span-type"] == "utterance", "end"] = (
        segments_df.loc[segments_df["span-type"] == "utterance", "segment-text"].str.len())

    # initialize texts
    texts = []
    mention_spans = []
    utterance_spans = []

    # initialize a new sequence
    current_block_text = ""
    current_block_text_size = 0
    current_block_n_tokens = 0
    current_block_n_segments = 0
    current_block_mention_spans_arrs = []
    current_block_utterance_spans_arrs = []

    # group segments by segment id
    segment_dfs = [segment_df for _, segment_df in segments_df.groupby("segment-order", sort=True)]
    i = 0

    # loop over segments
    while i < len(segment_dfs):
        segment_df = segment_dfs[i]
        n_segment_tokens = segment_df["n-tokens"].values[0]

        # add segment to sequence if it fits
        if current_block_n_tokens + n_segment_tokens < max_n_tokens_single_sentence:
            if current_block_n_segments:
                prefix = " "
                prefix_size = 1
            else:
                prefix = ""
                prefix_size = 0
            segment_text = segment_df["segment-text"].values[0]
            segment_spans_df = segment_df[["span-type", "start", "end"]]
            segment_spans_df.loc[:, ["start", "end"]] = segment_spans_df[["start", "end"]].astype(int)
            current_block_text += prefix + segment_text
            current_block_n_tokens += n_segment_tokens # assuming whitespace is not a token
            current_block_n_segments += 1
            segment_spans_df.loc[:, ["start", "end"]] += prefix_size + current_block_text_size
            current_block_text_size += prefix_size + len(segment_text)
            segment_mention_spans_arr = segment_spans_df.loc[segment_spans_df["span-type"] == "mention",
                                                             ["start", "end"]].to_numpy()
            segment_utterance_spans_arr = segment_spans_df.loc[segment_spans_df["span-type"] == "utterance",
                                                               ["start", "end"]].to_numpy()
            current_block_mention_spans_arrs.append(segment_mention_spans_arr)
            current_block_utterance_spans_arrs.append(segment_utterance_spans_arr)
            i += 1
        else:
            if current_block_n_segments:
                # add sequence to texts if it contains at least one segment
                j = len(texts)
                texts.append(current_block_text)
                current_block_mention_spans_arr = np.concatenate(current_block_mention_spans_arrs, axis=0)
                current_block_utterance_spans_arr = np.concatenate(current_block_utterance_spans_arrs, axis=0)
                for start, end in current_block_mention_spans_arr:
                    mention_spans.append([j, start, end])
                for start, end in current_block_utterance_spans_arr:
                    utterance_spans.append([j, start, end])

                # reinitialize a new sequence
                current_block_text = ""
                current_block_text_size = 0
                current_block_n_tokens = 0
                current_block_n_segments = 0
                current_block_mention_spans_arrs = []
                current_block_utterance_spans_arrs = []
            else:
                # segment is too big to fit, further segment it into subsegments
                segment_text = segment_df["segment-text"].values[0]
                segment_spans_df = segment_df[["span-type", "start", "end"]].dropna(axis=0)
                segment_spans_df[["start", "end"]] = segment_spans_df[["start", "end"]].astype(int)
                offsets_mapping = tokenizer(segment_text, return_offsets_mapping=True,
                                            add_special_tokens=False).offset_mapping
                start_token_id = 0

                # loop over sub segments
                while start_token_id < len(offsets_mapping):
                    end_token_id = start_token_id + max_n_tokens_single_sentence
    
                    # keep decreasing end token id until we reach the end of a full word
                    while ((end_token_id > start_token_id) 
                           and (end_token_id < len(offsets_mapping))
                           and (offsets_mapping[end_token_id - 1][1] == offsets_mapping[end_token_id][0])):
                        end_token_id -= 1

                    # add sub segment sequence
                    sub_segment_offsets_mapping = offsets_mapping[start_token_id: end_token_id]
                    sub_segment_text_start = sub_segment_offsets_mapping[0][0]
                    sub_segment_text_end = sub_segment_offsets_mapping[-1][1]
                    sub_segment_text = segment_text[sub_segment_text_start: sub_segment_text_end]
                    sub_segment_spans_df = segment_spans_df[(segment_spans_df["start"] >= sub_segment_text_start)
                                                            & (segment_spans_df["end"] <= sub_segment_text_end)]
                    sub_segment_spans_df.loc[:, ["start", "end"]] -= sub_segment_text_start
                    k = len(texts)
                    texts.append(sub_segment_text)
                    for _, row in sub_segment_spans_df.iterrows():
                        if row["span-type"] == "mention":
                            mention_spans.append([k, row["start"], row["end"]])
                        else:
                            utterance_spans.append([k, row["start"], row["end"]])
                    start_token_id = end_token_id
                i += 1

    # add the last block if it is not empty
    if current_block_n_segments:
        j = len(texts)
        texts.append(current_block_text)
        current_block_mention_spans_arr = np.concatenate(current_block_mention_spans_arrs, axis=0)
        current_block_utterance_spans_arr = np.concatenate(current_block_utterance_spans_arrs, axis=0)
        for start, end in current_block_mention_spans_arr:
            mention_spans.append([j, start, end])
        for start, end in current_block_utterance_spans_arr:
            utterance_spans.append([j, start, end])

    # find how many mention and utterance spans was not encoded
    n_mentions_not_in_blocks = n_mentions - len(mention_spans)
    n_utterances_not_in_blocks = n_utterances - len(utterance_spans)

    # tokenize texts
    tokenizer_output = tokenizer(texts, return_offsets_mapping=True, padding="max_length", return_tensors="pt",
                                 return_special_tokens_mask=True)
    token_ids = tokenizer_output.input_ids
    offsets_mapping = tokenizer_output.offset_mapping
    special_tokens_mask = tokenizer_output.special_tokens_mask
    n_blocks, n_tokens_sequence = token_ids.shape
    n_tokens = n_blocks * n_tokens_sequence
    mention_mask = torch.zeros((len(mention_spans), n_tokens), dtype=float)
    utterance_mask = torch.zeros((len(utterance_spans), n_tokens), dtype=float)
    n_mentions_not_matched, n_utterances_not_matched = 0, 0

    # match text spans to tokenizer offsets
    for i, (j, start, end) in enumerate(mention_spans):
        start_indices = torch.nonzero((offsets_mapping[j, :, 0] <= start) & (special_tokens_mask[j] == 0)).flatten()
        end_indices = torch.nonzero((offsets_mapping[j, :, 1] >= end) & (special_tokens_mask[j] == 0)).flatten()
        if len(start_indices) and len(end_indices) and start_indices.max() <= end_indices.min():
            p, q = start_indices.max(), end_indices.min()
            mention_mask[i, j * n_tokens_sequence + p: j * n_tokens_sequence + q + 1] = 1
        else:
            n_mentions_not_matched += 1

    # match text spans to tokenizer offsets
    for i, (j, start, end) in enumerate(utterance_spans):
        start_indices = torch.nonzero((offsets_mapping[j, :, 0] <= start) & (special_tokens_mask[j] == 0)).flatten()
        end_indices = torch.nonzero((offsets_mapping[j, :, 1] >= end) & (special_tokens_mask[j] == 0)).flatten()
        if len(start_indices) and len(end_indices) and start_indices.max() <= end_indices.min():
            p, q = start_indices.max(), end_indices.min()
            utterance_mask[i, j * n_tokens_sequence + p: j * n_tokens_sequence + q + 1] = 1
        else:
            n_utterances_not_matched += 1

    # remove non-matched spans
    mention_mask = mention_mask[(mention_mask == 1).any(axis=1)]
    utterance_mask = utterance_mask[(utterance_mask == 1).any(axis=1)]

    if n_mentions_not_in_blocks or n_mentions_not_matched or n_utterances_not_in_blocks or n_utterances_not_matched:
        logging.warn(f"{n_mentions} mentions, {n_mentions_not_in_blocks} mentions not in encoded text, "
              f"{n_mentions_not_matched} mentions not mapped to tokens")
        logging.warn(f"{n_utterances} utterances, {n_utterances_not_in_blocks} utterances not in encoded text, "
              f"{n_utterances_not_matched} utterances not mapped to tokens")

    # convert 0 to -inf, 1 to 0
    mention_mask = torch.log(mention_mask)
    utterance_mask = torch.log(utterance_mask)
    return token_ids, mention_mask, utterance_mask