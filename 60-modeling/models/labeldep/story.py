"""Find character representations conditioned on the label representations from complete story"""
from models.labeldep.base import LabelDependent
import utils

import torch

class StoryLabelDependent(LabelDependent):
    """Subclass of LabelDependent. Find character representations from complete story"""

    def name_representation(self, story_embeddings, names_idx, label_embeddings):
        """Create label-dependent name representations

        Input
        =====
            story_embeddings = [n_blocks, seqlen, hidden_size]
            names_idx = [n_characters, 2]
            label_embeddings = [n_labels, hidden_size]

        Output
        ======
            name_label_embeddings = [n_characters, n_labels, hidden_size]
        """
        device = next(self.parameters()).device
        dtype = names_idx.dtype
        seqlen = story_embeddings.shape[1]
        names_score = self.name_weights(story_embeddings).squeeze(dim=2) # [b, l]
        names_score = names_score.unsqueeze(dim=1) # [b, 1, l]
        sequence = torch.arange(seqlen, dtype=dtype, device=device).reshape(1, seqlen) # [1, l]
        names_mask = ((names_idx[:, 0].reshape(-1, 1) <= sequence)
                      & (names_idx[:, 1].reshape(-1, 1) > sequence)) # [k, l]
        names_mask = torch.log(names_mask) # [k, l]
        names_mask = names_mask.unsqueeze(dim=0) # [1, k, l]
        names_attn = torch.softmax(names_score + names_mask, dim=2) # [b, k, l]
        name_embeddings = torch.matmul(names_attn, story_embeddings) # [b, k, d]
        name_embeddings = name_embeddings.permute(1, 0, 2) # [k, b, d]
        label_embeddings = label_embeddings.unsqueeze(dim=0) # [1, t, d]
        name_label_score = torch.matmul(label_embeddings, name_embeddings.permute(0, 2, 1)) # [k, t, b]
        name_label_attn = torch.softmax(name_label_score, dim=2) # [k, t, b]
        name_label_embeddings = torch.matmul(name_label_attn, name_embeddings) # [k, t, d]
        label_embeddings = label_embeddings.squeeze(dim=0) # [t, d]
        return name_label_embeddings

    def mention_representation(self, story_embeddings, mentions_idx, label_embeddings):
        """Create label-dependent mention representations

        Input
        -----
            story_embeddings = [n_blocks, seqlen, hidden_size]
            mentions_idx = [n_characters, n_mentions, 2]
            label_embeddings = [n_labels, hidden_size]

        Output
        ------
            mention_label_embeddings = [n_characters, n_labels, hidden_size]
        """
        device = next(self.parameters()).device
        dtype_idx = mentions_idx.dtype
        dtype_embeddings = story_embeddings.dtype
        n_blocks, seqlen, hidden_size = story_embeddings.shape
        n_characters, n_mentions, _ = mentions_idx.shape
        n_labels = label_embeddings.shape[0]
        totlen = n_blocks * seqlen
        if n_mentions == 0:
            return torch.zeros((n_characters, n_labels, hidden_size), dtype=dtype_embeddings, device=device)
        sequence = torch.arange(totlen, dtype=dtype_idx, device=device).reshape(1, 1, totlen) # [1, 1, L]
        mentions_mask = ((mentions_idx[:,:,0].reshape(-1, n_mentions, 1) <= sequence)
                         & (mentions_idx[:,:,1].reshape(-1, n_mentions, 1) > sequence)) # [k, m, L]
        mentions_mask = torch.log(mentions_mask) # [k, m, L]
        story_embeddings = story_embeddings.reshape(1, totlen, hidden_size) # [1, L, d]
        mentions_score = self.mention_weights(story_embeddings).squeeze(dim=1) # [L]
        mentions_score = mentions_score.reshape(1, 1, totlen) # [1, 1, L]
        mentions_attn = utils.masked_softmax(mentions_score, mentions_mask) # [k, m, L]
        mention_embeddings = torch.matmul(mentions_attn, story_embeddings) # [k, m, d]
        label_embeddings = label_embeddings.unsqueeze(dim=0) # [1, t, d]
        mention_label_score = torch.matmul(label_embeddings, mention_embeddings.permute(0, 2, 1)) # [k, t, m]
        character_mentions_mask = torch.any(mentions_idx != 0, dim=2) # [k, m]
        character_mentions_mask = torch.log(character_mentions_mask) # [k, m]
        character_mentions_mask = character_mentions_mask.unsqueeze(dim=1) # [k, 1, m]
        mention_label_attn = utils.masked_softmax(mention_label_score, character_mentions_mask) # [k, t, m]
        mention_label_embeddings = torch.matmul(mention_label_attn, mention_embeddings) # [k, t, d]
        label_embeddings = label_embeddings.squeeze(dim=0) # [t, d]
        story_embeddings = story_embeddings.reshape(n_blocks, seqlen, hidden_size) # [b, l, d]
        return mention_label_embeddings

    def utterance_representation(self, story_embeddings, utterances_idx, label_embeddings):
        """Create label-dependent utterance representations

        Input
        -----
            story_embeddings = [n_blocks, seqlen, hidden_size]
            utterances_ids = [n_characters, n_utterances, 2]
            label_embeddings = [n_labels, hidden_size]

        Output
        ------
            utterance_label_embeddings = [n_characters, n_labels, hidden_size]
        """
        device = next(self.parameters()).device
        dtype_idx = utterances_idx.dtype
        dtype_embeddings = story_embeddings.dtype
        n_blocks, seqlen, hidden_size = story_embeddings.shape
        n_characters, n_utterances, _ = utterances_idx.shape
        n_labels = label_embeddings.shape[0]
        totlen = n_blocks * seqlen
        if n_utterances == 0:
            return torch.zeros((n_characters, n_labels, hidden_size), dtype=dtype_embeddings, device=device)
        story_embeddings = story_embeddings.reshape(totlen, hidden_size) # [L, d]
        utterance_score = torch.matmul(label_embeddings, story_embeddings.T) # [t, L]
        sequence = torch.arange(totlen, dtype=dtype_idx, device=device).reshape(1, 1, totlen) # [1, 1, L]
        utterances_mask = ((utterances_idx[:,:,0].reshape(-1, n_utterances, 1) <= sequence)
                           & (utterances_idx[:,:,1].reshape(-1, n_utterances, 1) > sequence)) # [k, u, L]
        utterances_mask = torch.any(utterances_mask, dim=1) # [k, L]
        utterances_mask = torch.log(utterances_mask) # [k, L]
        utterances_mask = utterances_mask.unsqueeze(dim=1) # [k, 1, L]
        utterance_score = utterance_score.unsqueeze(dim=0) # [1, t, L]
        utterance_attn = utils.masked_softmax(utterance_score, utterances_mask) # [k, t, L]
        story_embeddings = story_embeddings.unsqueeze(dim=0) # [1, L, d]
        utterance_embeddings = torch.matmul(utterance_attn, story_embeddings) # [k, t, d]
        story_embeddings = story_embeddings.reshape(n_blocks, seqlen, hidden_size) # [b, l, d]
        return utterance_embeddings

    def forward(self, story_embeddings, names_idx, mentions_idx, utterances_idx, label_embeddings):
        """Create character representations conditioned on the labels

        Input
        -----
            story_embeddings: [n_blocks, seqlen, hidden_size]
            names_idx: [n_characters, 2]
            mentions_idx: [n_characters, n_mentions, 2]
            utterances_idx: [n_characters, n_utterances, 2]
            label_embeddings: [n_labels, hidden_size]

        Output
        ------
            character_embeddings: [n_characters, n_labels, hidden_size]
        """
        name_embeddings = self.name_representation(story_embeddings, names_idx, label_embeddings)
        if self.use_mentions:
            mention_embeddings = self.mention_representation(story_embeddings, mentions_idx, label_embeddings)
        if self.use_utterances:
            utterance_embeddings = self.utterance_representation(story_embeddings, utterances_idx, label_embeddings)
        if self.use_mentions and self.use_utterances:
            character_embeddings = self.combine_embeddings(name_embeddings, mention_embeddings, utterance_embeddings)
        elif self.use_mentions:
            character_embeddings = self.combine_embeddings(name_embeddings, mention_embeddings)
        elif self.use_utterances:
            character_embeddings = self.combine_embeddings(name_embeddings, utterance_embeddings)
        else:
            character_embeddings = name_embeddings
        return character_embeddings