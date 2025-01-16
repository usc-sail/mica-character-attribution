"""Tensorize story segments"""
from absl import logging
import collections
import numpy as np
import pandas as pd
import torch

def create_tensors(tokenizer, imdb_characters, segments_df, mentions_df, utterances_df, add_prefix_space=False):
    """
    Tensorize the data for training and testing. The data will be of the form
    [CLS] ch1 ch2 ... chk [SEP] t1 t2 ... tn [SEP]. ch1, ch2, ... chk are k character names. t1, t2, ..., tn are tokens
    of the story segment. The output masks (mentions, utterances, and name masks) will be -inf/0 based.

    Input
    -----
        tokenizer = huggingface model tokenizer
        imdb_characters = list[str] of imdb character names
        segments_df = dataframe of segments
        mentions_df = dataframe of character mention spans
        utterances_df = dataframe of character utterance spans

    Output
    ------
        story_token_ids = long tensor [n_blocks, max_sequence_length]
        names_idx = float tensor [n_characters, 2]
        mentions_idx = float tensor [n_mentions, 2]
        utterances_idx = float tensor [n_utterances, 2]
        mention_character_ids = long tensor [n_mentions]
        utterance_character_ids = long tensor [n_utterances]

    Therefore, the total size is O(n_blocks x max_sequence_length + n_characters + n_mentions + n_utterances)
    """
    # create the first sentence of the sequence
    max_n_tokens_sentence_pair = tokenizer.max_len_sentences_pair
    imdb_characters_sentence = " ".join(imdb_characters)
    imdb_character_spans = []
    c = 0
    for imdb_character in imdb_characters:
        imdb_character_spans.append([c, c + len(imdb_character)])
        c += len(imdb_character) + 1
    tokenizer_output = tokenizer(imdb_characters_sentence, return_offsets_mapping=True, add_special_tokens=True,
                                 return_special_tokens_mask=True)
    n_tokens_imdb_characters_sentence = sum([m == 0 for m in tokenizer_output.special_tokens_mask])
    n_tokens_first_sentence = len(tokenizer_output.input_ids)
    imdb_character_to_id = dict([(imdb_character, i) for i, imdb_character in enumerate(imdb_characters)])

    # create name mask
    names_idx = [[0, 0] for _ in range(len(imdb_characters))]
    offsets_mapping = tokenizer_output.offset_mapping
    special_tokens_mask = tokenizer_output.special_tokens_mask
    i = 0
    j = 0
    while j < len(imdb_characters) and i < len(offsets_mapping):
        if special_tokens_mask[i] == 0 and offsets_mapping[i][0] == imdb_character_spans[j][0]:
            k = i
            while k < len(offsets_mapping) and (special_tokens_mask[k] == 1 
                                                or offsets_mapping[k][1] != imdb_character_spans[j][1]):
                k += 1
            names_idx[j] = [i, k + 1]
            j += 1
            i = k
        else:
            i += 1
    names_idx = torch.tensor(names_idx)

    # join segments dataframe with the mention and utterance spans dataframes
    segments_df["n-tokens"] = segments_df["segment-text"].apply(
        lambda text: len(tokenizer((" " if add_prefix_space else "") + text, add_special_tokens=False).input_ids))
    segments_df["segment-order"] = np.arange(len(segments_df)).tolist()
    mentions_df["span-type"] = "mention"
    utterances_df["span-type"] = "utterance"
    utterances_df["start"] = 0
    spans_df = pd.concat([mentions_df, utterances_df], axis=0)
    spans_df = spans_df[spans_df["imdb-character"].isin(imdb_characters)]
    spans_df["imdb-character-id"] = spans_df["imdb-character"].apply(lambda ch: imdb_character_to_id[ch])
    n_mentions = (spans_df["span-type"] == "mention").sum()
    n_utterances = (spans_df["span-type"] == "utterance").sum()
    segments_df = segments_df.merge(spans_df, how="left", on="segment-id")
    segments_df.loc[segments_df["span-type"] == "utterance", "end"] = (
        segments_df.loc[segments_df["span-type"] == "utterance", "segment-text"].str.len())

    # initialize text pairs
    text_pairs = []
    mention_spans = []
    utterance_spans = []

    # initialize a new sequence
    current_block_text = ""
    current_block_text_size = 0
    current_block_n_tokens = n_tokens_imdb_characters_sentence
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
        if current_block_n_tokens + n_segment_tokens < max_n_tokens_sentence_pair:
            if add_prefix_space:
                prefix = " "
                prefix_size = 1
            else:
                if current_block_n_segments:
                    prefix = " "
                    prefix_size = 1
                else:
                    prefix = ""
                    prefix_size = 0
            segment_text = segment_df["segment-text"].values[0]
            segment_spans_df = segment_df[["span-type", "imdb-character-id", "start", "end"]].dropna(axis=0)
            segment_spans_df[["imdb-character-id", "start", "end"]] = (
                segment_spans_df[["imdb-character-id", "start", "end"]].astype(int))
            current_block_text += prefix + segment_text
            current_block_n_tokens += n_segment_tokens # assuming whitespace is not a token
            current_block_n_segments += 1
            segment_spans_df.loc[:, ["start", "end"]] += prefix_size + current_block_text_size
            current_block_text_size += prefix_size + len(segment_text)
            segment_mention_spans_arr = segment_spans_df.loc[segment_spans_df["span-type"] == "mention",
                                                             ["imdb-character-id", "start", "end"]].to_numpy()
            segment_utterance_spans_arr = segment_spans_df.loc[segment_spans_df["span-type"] == "utterance",
                                                               ["imdb-character-id", "start", "end"]].to_numpy()
            current_block_mention_spans_arrs.append(segment_mention_spans_arr)
            current_block_utterance_spans_arrs.append(segment_utterance_spans_arr)
            # actual = len(tokenizer(current_block_text, add_special_tokens=False).input_ids)
            # logging.info(f"calc = {current_block_n_tokens - n_tokens_imdb_characters_sentence}, actual = {actual}")
            i += 1
        else:
            if current_block_n_segments:
                # add sequence to text pairs if it contains at least one segment
                j = len(text_pairs)
                text_pairs.append([imdb_characters_sentence, current_block_text])
                current_block_mention_spans_arr = np.concatenate(current_block_mention_spans_arrs, axis=0)
                current_block_utterance_spans_arr = np.concatenate(current_block_utterance_spans_arrs, axis=0)
                for character_id, start, end in current_block_mention_spans_arr:
                    mention_spans.append([j, character_id, start, end])
                for character_id, start, end in current_block_utterance_spans_arr:
                    utterance_spans.append([j, character_id, start, end])

                # reinitialize a new sequence
                current_block_text = ""
                current_block_text_size = 0
                current_block_n_tokens = n_tokens_imdb_characters_sentence
                current_block_n_segments = 0
                current_block_mention_spans_arrs = []
                current_block_utterance_spans_arrs = []
            else:
                # segment is too big to fit, further segment it into subsegments
                segment_text = segment_df["segment-text"].values[0]
                segment_spans_df = segment_df[["span-type", "imdb-character-id", "start", "end"]].dropna(axis=0)
                segment_spans_df[["imdb-character-id", "start", "end"]] = (
                    segment_spans_df[["imdb-character-id", "start", "end"]].astype(int))
                offsets_mapping = tokenizer(segment_text, return_offsets_mapping=True,
                                            add_special_tokens=False).offset_mapping
                n_tokens_sub_segment = max_n_tokens_sentence_pair - n_tokens_imdb_characters_sentence
                start_token_id = 0

                # loop over sub segments
                while start_token_id < len(offsets_mapping):
                    end_token_id = start_token_id + n_tokens_sub_segment
    
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
                    k = len(text_pairs)
                    text_pairs.append([imdb_characters_sentence, sub_segment_text])
                    for _, row in sub_segment_spans_df.iterrows():
                        if row["span-type"] == "mention":
                            mention_spans.append([k, row["imdb-character-id"], row["start"], row["end"]])
                        else:
                            utterance_spans.append([k, row["imdb-character-id"], row["start"], row["end"]])
                    start_token_id = end_token_id
                i += 1

    # add the last block if it is not empty
    if current_block_n_segments:
        j = len(text_pairs)
        text_pairs.append([imdb_characters_sentence, current_block_text])
        current_block_mention_spans_arr = np.concatenate(current_block_mention_spans_arrs, axis=0)
        current_block_utterance_spans_arr = np.concatenate(current_block_utterance_spans_arrs, axis=0)
        for character_id, start, end in current_block_mention_spans_arr:
            mention_spans.append([j, character_id, start, end])
        for character_id, start, end in current_block_utterance_spans_arr:
            utterance_spans.append([j, character_id, start, end])

    # find how many mention and utterance spans was not encoded
    n_mentions_not_in_blocks = n_mentions - len(mention_spans)
    n_utterances_not_in_blocks = n_utterances - len(utterance_spans)

    # tokenize text pairs
    tokenizer_output = tokenizer(text_pairs, return_offsets_mapping=True, padding="max_length", return_tensors="pt",
                                 return_special_tokens_mask=True, return_attention_mask=True)
    token_ids = tokenizer_output.input_ids
    attention_mask = tokenizer_output.attention_mask
    offsets_mapping = tokenizer_output.offset_mapping
    special_tokens_mask = tokenizer_output.special_tokens_mask
    _, n_tokens_sequence = token_ids.shape
    second_sentence_mask = torch.arange(n_tokens_sequence) >= n_tokens_first_sentence
    mentions_idx = [[0, 0] for _ in range(len(mention_spans))]
    utterances_idx = [[0, 0] for _ in range(len(utterance_spans))]
    mention_character_ids = []
    utterance_character_ids = []
    n_mentions_not_matched, n_utterances_not_matched = 0, 0

    # match text spans to tokenizer offsets
    for i, (j, k, start, end) in enumerate(mention_spans):
        start_indices = torch.nonzero((offsets_mapping[j, :, 0] <= start) & (special_tokens_mask[j] == 0)
                                       & second_sentence_mask).flatten()
        end_indices = torch.nonzero((offsets_mapping[j, :, 1] >= end) & (special_tokens_mask[j] == 0)
                                     & second_sentence_mask).flatten()
        if len(start_indices) and len(end_indices) and start_indices.max() <= end_indices.min():
            p, q = start_indices.max(), end_indices.min()
            mentions_idx[i] = [j * n_tokens_sequence + p, j * n_tokens_sequence + q + 1]
            mention_character_ids.append(k)
        else:
            n_mentions_not_matched += 1

    # match text spans to tokenizer offsets
    for i, (j, k, start, end) in enumerate(utterance_spans):
        start_indices = torch.nonzero((offsets_mapping[j, :, 0] <= start) & (special_tokens_mask[j] == 0)
                                       & second_sentence_mask).flatten()
        end_indices = torch.nonzero((offsets_mapping[j, :, 1] >= end) & (special_tokens_mask[j] == 0)
                                     & second_sentence_mask).flatten()
        if len(start_indices) and len(end_indices) and start_indices.max() <= end_indices.min():
            p, q = start_indices.max(), end_indices.min()
            utterances_idx[i] = [j * n_tokens_sequence + p, j * n_tokens_sequence + q + 1]
            utterance_character_ids.append(k)
        else:
            n_utterances_not_matched += 1

    # remove non-matched mention spans
    mentions_idx = torch.tensor(mentions_idx)
    mention_character_ids = torch.tensor(mention_character_ids)
    if len(mention_spans) > 0:
        mention_character_ids = mention_character_ids[(mentions_idx != 0).any(axis=1)]
        mentions_idx = mentions_idx[(mentions_idx != 0).any(axis=1)]

        # convert mentions idx into characters x mentions x spans tensor
        max_n_mentions = max(collections.Counter(mention_character_ids.tolist()).values())
        mentions_idx2 = torch.zeros((len(imdb_characters), max_n_mentions, 2), dtype=int)
        for k in range(len(imdb_characters)):
            character_mentions_idx = mentions_idx[mention_character_ids == k]
            n = len(character_mentions_idx)
            mentions_idx2[k, :n] = character_mentions_idx
        assert ((mentions_idx2 != 0).any(dim=2).sum(dim=1) == mentions_idx2.shape[1]).any()
    else:
        mentions_idx2 = torch.empty((len(imdb_characters), 0, 2), dtype=int)

    # remove non-matched utterance spans
    utterances_idx = torch.tensor(utterances_idx)
    utterance_character_ids = torch.tensor(utterance_character_ids)
    if len(utterance_spans) > 0:
        utterance_character_ids = utterance_character_ids[(utterances_idx != 0).any(axis=1)]
        utterances_idx = utterances_idx[(utterances_idx != 0).any(axis=1)]

        # convert utterances idx into characters x mentions x spans tensor
        max_n_utterances = max(collections.Counter(utterance_character_ids.tolist()).values())
        utterances_idx2 = torch.zeros((len(imdb_characters), max_n_utterances, 2), dtype=int)
        for k in range(len(imdb_characters)):
            character_utterances_idx = utterances_idx[utterance_character_ids == k]
            n = len(character_utterances_idx)
            utterances_idx2[k, :n] = character_utterances_idx
        assert ((utterances_idx2 != 0).any(dim=2).sum(dim=1) == utterances_idx2.shape[1]).any()
    else:
        utterances_idx2 = torch.empty((len(imdb_characters), 0, 2), dtype=int)

    if n_mentions_not_in_blocks or n_mentions_not_matched or n_utterances_not_in_blocks or n_utterances_not_matched:
        logging.warn(f"{n_mentions} mentions, {n_mentions_not_in_blocks} mentions not in encoded text, "
              f"{n_mentions_not_matched} mentions not mapped to tokens")
        logging.warn(f"{n_utterances} utterances, {n_utterances_not_in_blocks} utterances not in encoded text, "
              f"{n_utterances_not_matched} utterances not mapped to tokens")

    return token_ids, attention_mask, mentions_idx2, utterances_idx2, names_idx