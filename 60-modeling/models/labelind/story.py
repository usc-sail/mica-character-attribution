"""Find character representations independent of label representations from complete story"""
from models.labelind.base import LabelIndependent
import utils

import torch

class StoryLabelIndependent(LabelIndependent):
    """Subclass of LabelIndependent. Find character representations from complete story"""

    def name_representation(self, story_embeddings, names_idx):
        """Create label-independent name representations

        Input
        =====
            story_embeddings = [n_blocks, seqlen, hidden_size]
            names_idx = [n_characters, 2]

        Output
        ======
            name_embeddings = [n_characters, hidden_size]
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
        name_seq_score = self.name_seq_weights(name_embeddings).squeeze(dim=2) # [k, b]
        name_seq_attn = torch.softmax(name_seq_score, dim=1) # [k, b]
        name_seq_attn = name_seq_attn.unsqueeze(dim=1) # [k, 1, b]
        name_seq_embeddings = torch.matmul(name_seq_attn, name_embeddings).squeeze(dim=1) # [k, d]
        return name_seq_embeddings

    def mention_representation(self, story_embeddings, mentions_idx):
        """Create label-independent mention representations

        Input
        -----
            story_embeddings = [n_blocks, seqlen, hidden_size]
            mentions_idx = [n_characters, n_mentions, 2]

        Output
        ------
            mention_embeddings = [n_characters, hidden_size]
        """
        device = next(self.parameters()).device
        dtype_idx = mentions_idx.dtype
        dtype_embeddings = story_embeddings.dtype
        n_blocks, seqlen, hidden_size = story_embeddings.shape
        n_characters, n_mentions, _ = mentions_idx.shape
        totlen = n_blocks * seqlen
        if n_mentions == 0:
            return torch.zeros((n_characters, hidden_size), dtype=dtype_embeddings, device=device)
        sequence = torch.arange(totlen, dtype=dtype_idx, device=device).reshape(1, 1, totlen) # [1, 1, L]
        mentions_mask = ((mentions_idx[:,:,0].reshape(-1, n_mentions, 1) <= sequence)
                         & (mentions_idx[:,:,1].reshape(-1, n_mentions, 1) > sequence)) # [k, m, L]
        mentions_mask = torch.log(mentions_mask) # [k, m, L]
        story_embeddings = story_embeddings.reshape(1, totlen, hidden_size) # [1, L, d]
        mentions_score = self.mention_weights(story_embeddings).squeeze(dim=1) # [L]
        mentions_score = mentions_score.reshape(1, 1, totlen) # [1, 1, L]
        mentions_attn = utils.masked_softmax(mentions_score, mentions_mask) # [k, m, L]
        mention_embeddings = torch.matmul(mentions_attn, story_embeddings) # [k, m, d]
        mentions_seq_score = self.mention_seq_weights(mention_embeddings).squeeze(dim=2) # [k, m]
        character_mentions_mask = torch.any(mentions_idx != 0, dim=2) # [k, m]
        character_mentions_mask = torch.log(character_mentions_mask) # [k, m]
        mentions_seq_attn = utils.masked_softmax(mentions_seq_score, character_mentions_mask) # [k, m]
        mentions_seq_attn = mentions_seq_attn.unsqueeze(dim=1) # [k, 1, m]
        mentions_seq_embeddings = torch.matmul(mentions_seq_attn, mention_embeddings).squeeze(dim=1) # [k, d]
        story_embeddings = story_embeddings.reshape(n_blocks, seqlen, hidden_size) # [b, l, d]
        return mentions_seq_embeddings

    def utterance_representation(self, story_embeddings, utterances_idx):
        """Create label-independent utterance representations

        Input
        -----
            story_embeddings = [n_blocks, seqlen, hidden_size]
            utterances_ids = [n_characters, n_utterances, 2]

        Output
        ------
            utterance_embeddings = [n_characters, hidden_size]
        """
        device = next(self.parameters()).device
        dtype_idx = utterances_idx.dtype
        dtype_embeddings = story_embeddings.dtype
        n_blocks, seqlen, hidden_size = story_embeddings.shape
        n_characters, n_utterances, _ = utterances_idx.shape
        totlen = n_blocks * seqlen
        if n_utterances == 0:
            return torch.zeros((n_characters, hidden_size), dtype=dtype_embeddings, device=device)
        story_embeddings = story_embeddings.reshape(totlen, hidden_size) # [L, d]
        utterance_score = self.utterance_seq_weights(story_embeddings).squeeze(dim=1) # [L]
        sequence = torch.arange(totlen, dtype=dtype_idx, device=device).reshape(1, 1, totlen) # [1, 1, L]
        utterances_mask = ((utterances_idx[:,:,0].reshape(-1, n_utterances, 1) <= sequence)
                            & (utterances_idx[:,:,1].reshape(-1, n_utterances, 1) > sequence)) # [k, u, L]
        utterances_mask = torch.any(utterances_mask, dim=1) # [k, L]
        utterances_mask = torch.log(utterances_mask) # [k, L]
        utterance_score = utterance_score.unsqueeze(dim=0) # [1, L]
        utterance_attn = utils.masked_softmax(utterance_score, utterances_mask) # [k, L]
        utterance_embeddings = torch.matmul(utterance_attn, story_embeddings) # [k, d]
        story_embeddings = story_embeddings.reshape(n_blocks, seqlen, hidden_size) # [b, l, d]
        return utterance_embeddings

    def forward(self, story_embeddings, names_idx, mentions_idx, utterances_idx):
        """Create character representations independent of label

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
        name_embeddings = self.name_representation(story_embeddings, names_idx)
        if self.use_mentions:
            mention_embeddings = self.mention_representation(story_embeddings, mentions_idx)
        if self.use_utterances:
            utterance_embeddings = self.utterance_representation(story_embeddings, utterances_idx)
        if self.use_mentions and self.use_utterances:
            character_embeddings = self.combine_embeddings(name_embeddings, mention_embeddings, utterance_embeddings)
        elif self.use_mentions:
            character_embeddings = self.combine_embeddings(name_embeddings, mention_embeddings)
        elif self.use_utterances:
            character_embeddings = self.combine_embeddings(name_embeddings, utterance_embeddings)
        else:
            character_embeddings = name_embeddings
        return character_embeddings