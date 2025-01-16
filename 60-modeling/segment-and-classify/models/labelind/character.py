"""Find character representations independent of label representations from character segments"""
from models.labelind.base import LabelIndependent
import utils

import torch

class CharacterLabelIndependent(LabelIndependent):
    """Subclass of LabelIndependent. Find character representations from character segments"""

    def name_representation(self, segment_embeddings, segment_mask, names_idx):
        """Create name representations

        Input
        -----
            segment_embeddings: [n_characters, n_blocks, seqlen, hidden_size]
            segment_mask: [n_characters, n_blocks]
            names_idx: [n_characters, 2]

        Output
        ------
            name_embeddings: [n_characters, hidden_size]
        """
        names_score = self.name_weights(segment_embeddings).squeeze(dim=3) # [k, b, l]
        device = next(self.parameters()).device
        dtype = names_idx.dtype
        seqlen = segment_embeddings.shape[2]
        sequence = torch.arange(seqlen, dtype=dtype, device=device).reshape(1, seqlen) # [1, l]
        names_mask = ((names_idx[:, 0].reshape(-1, 1) <= sequence)
                      & (names_idx[:, 1].reshape(-1, 1) > sequence)) # [k, l]
        names_mask = torch.log(names_mask) # [k, l]
        names_mask = names_mask.unsqueeze(dim=1) # [k, 1, l]
        names_attn = torch.softmax(names_score + names_mask, dim=2) # [k, b, l]
        names_attn = names_attn.unsqueeze(dim=2) # [k, b, 1, l]
        name_embeddings = torch.matmul(names_attn, segment_embeddings).squeeze(dim=2) # [k, b, d]
        name_seq_score = self.name_seq_weights(name_embeddings).squeeze(dim=2) # [k, b]
        name_seq_attn = torch.softmax(name_seq_score + segment_mask, dim=1) # [k, b]
        name_seq_attn = name_seq_attn.unsqueeze(dim=1) # [k, 1, b]
        name_seq_embeddings = torch.matmul(name_seq_attn, name_embeddings).squeeze(dim=1) # [k, d]
        return name_seq_embeddings

    def mention_representation(self, segment_embeddings, mentions_idx):
        """Create mention representations

        Input
        -----
            segment_embeddings: [n_characters, n_blocks, seqlen, hidden_size]
            mentions_idx: [n_characters, n_mentions, 2]

        Output
        ------
            mention_embeddings: [n_characters, hidden_size]
        """
        device = next(self.parameters()).device
        dtype_idx = mentions_idx.dtype
        dtype_embeddings = segment_embeddings.dtype
        n_characters, n_blocks, seqlen, hidden_size = segment_embeddings.shape
        n_mentions = mentions_idx.shape[1]
        totlen = n_blocks * seqlen
        if n_mentions == 0:
            return torch.zeros((n_characters, hidden_size), dtype=dtype_embeddings, device=device)
        mentions_score = self.mention_weights(segment_embeddings).squeeze(dim=3) # [k, b, l]
        mentions_score = mentions_score.reshape(n_characters, 1, totlen) # [k, 1, L]
        sequence = torch.arange(totlen, dtype=dtype_idx, device=device).reshape(1, 1, totlen) # [1, 1, L]
        mentions_mask = ((mentions_idx[:,:,0].reshape(-1, n_mentions, 1) <= sequence)
                         & (mentions_idx[:,:,1].reshape(-1, n_mentions, 1) > sequence)) # [k, m, L]
        mentions_mask = torch.log(mentions_mask) # [k, m, L], all L = -inf
        mentions_attn = utils.masked_softmax(mentions_score, mentions_mask) # [k, m, L]
        segment_embeddings = segment_embeddings.reshape(n_characters, totlen, hidden_size) # [k, L, d]
        mentions_embeddings = torch.matmul(mentions_attn, segment_embeddings) # [k, m, d], all d = nan
        mentions_seq_score = self.mention_seq_weights(mentions_embeddings).squeeze(dim=2) # [k, m]
        character_mentions_mask = torch.any(mentions_idx != 0, dim=2) # [k, m]
        character_mentions_mask = torch.log(character_mentions_mask) # [k, m]
        mentions_seq_attn = utils.masked_softmax(mentions_seq_score, character_mentions_mask) # [k, m]
        mentions_seq_attn = mentions_seq_attn.unsqueeze(dim=1) # [k, 1, m]
        mentions_seq_embeddings = torch.matmul(mentions_seq_attn, mentions_embeddings).squeeze(dim=1) # [k, d]
        segment_embeddings = segment_embeddings.reshape(n_characters, n_blocks, seqlen, hidden_size) # [k, b, l, d]
        return mentions_seq_embeddings

    def utterance_representation(self, segment_embeddings, utterances_idx):
        """Create utterance respresentations

        Input
        -----
            segment_embeddings: [n_characters, n_blocks, seqlen, hidden_size]
            utterances_idx: [n_characters, n_utterances, 2]

        Output
        ------
            utterance_embeddings: [n_characters, hidden_size]
        """
        device = next(self.parameters()).device
        dtype_idx = utterances_idx.dtype
        dtype_embeddings = segment_embeddings.dtype
        n_characters, n_blocks, seqlen, hidden_size = segment_embeddings.shape
        n_utterances = utterances_idx.shape[1]
        totlen = n_blocks * seqlen
        if n_utterances == 0:
            return torch.zeros((n_characters, hidden_size), dtype=dtype_embeddings, device=device)
        segment_embeddings = segment_embeddings.reshape(n_characters, totlen, hidden_size) # [k, L, d]
        utterances_score = self.utterance_seq_weights(segment_embeddings).squeeze(dim=2) # [k, L]
        sequence = torch.arange(totlen, dtype=dtype_idx, device=device).reshape(1, 1, totlen) # [1, 1, L]
        utterances_mask = ((utterances_idx[:,:,0].reshape(-1, n_utterances, 1) <= sequence)
                           & (utterances_idx[:,:,1].reshape(-1, n_utterances, 1) > sequence)) # [k, u, L]
        utterances_mask = torch.any(utterances_mask, dim=1) # [k, L]
        utterances_mask = torch.log(utterances_mask) # [k, L]
        utterances_attn = utils.masked_softmax(utterances_score, utterances_mask) # [k, L]
        utterances_attn = utterances_attn.unsqueeze(dim=1) # [k, 1, L], for some k, all L = nan
        utterance_embeddings = torch.matmul(utterances_attn, segment_embeddings).squeeze(dim=1) # [k, d]
        segment_embeddings = segment_embeddings.reshape(n_characters, n_blocks, seqlen, hidden_size) # [k, b, l, d]
        return utterance_embeddings

    def forward(self, segment_embeddings, segment_mask, names_idx, mentions_idx, utterances_idx):
        """Create character representations independent of label representations

        Input
        -----
            segment_embeddings: [n_characters, n_blocks, seqlen, hidden_size]
            segment_mask: [n_characters, n_blocks]
            names_idx: [n_characters, 2]
            mentions_idx: [n_characters, n_mentions, 2]
            utterances_idx: [n_characters, n_utterances, 2]

        Output
        ------
            character_embeddings: [n_characters, hidden_size]
        """
        name_embeddings = self.name_representation(segment_embeddings, segment_mask, names_idx)
        if self.use_mentions:
            mention_embeddings = self.mention_representation(segment_embeddings, mentions_idx)
        if self.use_utterances:
            utterance_embeddings = self.utterance_representation(segment_embeddings, utterances_idx)
        if self.use_mentions and self.use_utterances:
            character_embeddings = self.combine_embeddings(name_embeddings, mention_embeddings, utterance_embeddings)
        elif self.use_mentions:
            character_embeddings = self.combine_embeddings(name_embeddings, mention_embeddings)
        elif self.use_utterances:
            character_embeddings = self.combine_embeddings(name_embeddings, utterance_embeddings)
        else:
            character_embeddings = name_embeddings
        return character_embeddings