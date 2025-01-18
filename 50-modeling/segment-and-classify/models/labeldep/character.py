"""Find character representations conditioned on the label representations from character segments"""
from models.labeldep.base import LabelDependent
import utils

import torch

class CharacterLabelDependent(LabelDependent):
    """Subclass of LabelDependent. Find character representations from character segments"""

    def name_representation(self, segment_embeddings, segment_mask, names_idx, label_embeddings):
        """Create label-dependent name representations

        Input
        -----
            segment_embeddings: [n_characters, n_blocks, seqlen, hidden_size]
            segment_mask: [n_characters, n_blocks]
            names_idx: [n_characters, 2]
            label_embeddings: [n_labels, hidden_size]

        Output
        ------
            name_label_embeddings: [n_characters, n_labels, hidden_size]
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
        label_embeddings = label_embeddings.unsqueeze(dim=0) # [1, t, d]
        name_label_score = torch.matmul(label_embeddings, name_embeddings.permute(0, 2, 1)) # [k, t, b]
        name_label_mask = segment_mask.unsqueeze(dim=1) # [k, 1, b]
        name_label_attn = torch.softmax(name_label_score + name_label_mask, dim=2) # [k, t, b]
        name_label_embeddings = torch.matmul(name_label_attn, name_embeddings) # [k, t, d]
        label_embeddings = label_embeddings.squeeze(dim=0) # [t, d]
        return name_label_embeddings

    def mention_representation(self, segment_embeddings, mentions_idx, label_embeddings):
        """Create label-dependent mention representations

        Input
        -----
            segment_embeddings: [n_characters, n_blocks, seqlen, hidden_size]
            mentions_idx: [n_characters, n_mentions, 2]
            label_embeddings: [n_labels, hidden_size]

        Output
        ------
            mention_label_embeddings: [n_characters, n_labels, hidden_size]
        """
        device = next(self.parameters()).device
        dtype_idx = mentions_idx.dtype
        dtype_embeddings = segment_embeddings.dtype
        n_characters, n_blocks, seqlen, hidden_size = segment_embeddings.shape
        n_mentions = mentions_idx.shape[1]
        n_labels = label_embeddings.shape[0]
        totlen = n_blocks * seqlen
        if n_mentions == 0:
            return torch.zeros((n_characters, n_labels, hidden_size), dtype=dtype_embeddings, device=device)
        mentions_score = self.mention_weights(segment_embeddings).squeeze(dim=3) # [k, b, l]
        mentions_score = mentions_score.reshape(n_characters, 1, totlen) # [k, 1, L]
        sequence = torch.arange(totlen, dtype=dtype_idx, device=device).reshape(1, 1, totlen) # [1, 1, L]
        mentions_mask = ((mentions_idx[:,:,0].reshape(-1, n_mentions, 1) <= sequence)
                         & (mentions_idx[:,:,1].reshape(-1, n_mentions, 1) > sequence)) # [k, m, L]
        mentions_mask = torch.log(mentions_mask) # [k, m, L], for some m, all L = -inf
        mentions_attn = utils.masked_softmax(mentions_score, mentions_mask) # [k, m, L]
        segment_embeddings = segment_embeddings.reshape(n_characters, totlen, hidden_size) # [k, L, d]
        mention_embeddings = torch.matmul(mentions_attn, segment_embeddings) # [k, m, d], for some, all d = nan
        label_embeddings = label_embeddings.unsqueeze(dim=0) # [1, t, d]
        mentions_label_score = torch.matmul(label_embeddings,
                                            mention_embeddings.permute(0, 2, 1)) # [k, t, m], for some m, all 0
        character_mentions_mask = torch.any(mentions_idx != 0, dim=2) # [k, m]
        character_mentions_mask = torch.log(character_mentions_mask) # [k, m], some m = -inf
        character_mentions_mask = character_mentions_mask.unsqueeze(dim=1) # [k, 1, m], some m = -inf, for some k, all m = -inf
        mentions_label_attn = utils.masked_softmax(mentions_label_score, character_mentions_mask) # [k, t, m]
        mention_label_embeddings = torch.matmul(mentions_label_attn, mention_embeddings) # [k, t, d]
        segment_embeddings = segment_embeddings.reshape(n_characters, n_blocks, seqlen, hidden_size) # [k, b, l, d]
        label_embeddings = label_embeddings.squeeze(dim=0) # [t, d]
        return mention_label_embeddings

    def utterance_representation(self, segment_embeddings, utterances_idx, label_embeddings):
        """Create label-dependent utterance respresentations

        Input
        -----
            segment_embeddings: [n_characters, n_blocks, seqlen, hidden_size]
            utterances_idx: [n_characters, n_utterances, 2]
            label_embeddings: [n_labels, hidden_size]

        Output
        ------
            utterance_label_embeddings: [n_characters, n_labels, hidden_size]
        """
        device = next(self.parameters()).device
        dtype_idx = utterances_idx.dtype
        dtype_embeddings = segment_embeddings.dtype
        n_characters, n_blocks, seqlen, hidden_size = segment_embeddings.shape
        n_utterances = utterances_idx.shape[1]
        n_labels = label_embeddings.shape[0]
        totlen = n_blocks * seqlen
        if n_utterances == 0:
            return torch.zeros((n_characters, n_labels, hidden_size), dtype=dtype_embeddings, device=device)
        segment_embeddings = segment_embeddings.reshape(n_characters, totlen, hidden_size) # [k, L, d]
        label_embeddings = label_embeddings.T.unsqueeze(dim=0) # [1, d, t]
        utterances_score = torch.matmul(segment_embeddings, label_embeddings) # [k, L, t]
        utterances_score = utterances_score.permute(0, 2, 1) # [k, t, L]
        sequence = torch.arange(totlen, dtype=dtype_idx, device=device).reshape(1, 1, totlen) # [1, 1, L]
        utterances_mask = ((utterances_idx[:,:,0].reshape(-1, n_utterances, 1) <= sequence)
                           & (utterances_idx[:,:,1].reshape(-1, n_utterances, 1) > sequence)) # [k, u, L]
        utterances_mask = torch.any(utterances_mask, dim=1) # [k, L]
        utterances_mask = torch.log(utterances_mask) # [k, L]
        utterances_mask = utterances_mask.unsqueeze(dim=1) # [k, 1, L]
        utterances_attn = utils.masked_softmax(utterances_score, utterances_mask) # [k, t, L]
        utterance_embeddings = torch.matmul(utterances_attn, segment_embeddings) # [k, t, d]
        segment_embeddings = segment_embeddings.reshape(n_characters, n_blocks, seqlen, hidden_size) # [k, b, l, d]
        label_embeddings = label_embeddings.squeeze(dim=0).T # [t, d]
        return utterance_embeddings

    def forward(self, segment_embeddings, segment_mask, names_idx, mentions_idx, utterances_idx, label_embeddings):
        """Create character representations conditioned on the labels

        Input
        -----
            segment_embeddings: [n_characters, n_blocks, seqlen, hidden_size]
            segment_mask: [n_characters, n_blocks]
            names_idx: [n_characters, 2]
            mentions_idx: [n_characters, n_mentions, 2]
            utterances_idx: [n_characters, n_utterances, 2]
            label_embeddings: [n_labels, hidden_size]

        Output
        ------
            character_embeddings: [n_characters, n_labels, hidden_size]
        """
        name_embeddings = self.name_representation(segment_embeddings, segment_mask, names_idx, label_embeddings)
        if self.use_mentions:
            mention_embeddings = self.mention_representation(segment_embeddings, mentions_idx, label_embeddings)
        if self.use_utterances:
            utterance_embeddings = self.utterance_representation(segment_embeddings, utterances_idx, label_embeddings)
        if self.use_mentions and self.use_utterances:
            character_embeddings = self.combine_embeddings(name_embeddings, mention_embeddings, utterance_embeddings)
        elif self.use_mentions:
            character_embeddings = self.combine_embeddings(name_embeddings, mention_embeddings)
        elif self.use_utterances:
            character_embeddings = self.combine_embeddings(name_embeddings, utterance_embeddings)
        else:
            character_embeddings = name_embeddings
        return character_embeddings