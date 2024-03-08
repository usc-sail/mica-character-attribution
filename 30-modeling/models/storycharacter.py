"""Model to find character representations from story"""

import torch
from torch import nn
from transformers import AutoModel

class StoryCharacter(nn.Module):

    def __init__(self, pretrained_model_name, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

    def forward(self, n_characters, story_input_ids, trope_input_ids, mentions_mask, utterances_mask):
        """
        story_input_ids = batch_size x max_sequence_length
        trope_input_ids = n_tropes x max_sequence_length
        mentions_mask = n_mentions x total_sequence_length
        utterances_mask = n_utterances x total_sequence_length
        """
        embeddings = self.encoder(story_input_ids).last_hidden_state
        