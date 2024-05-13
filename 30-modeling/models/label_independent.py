"""Find character representations from the name, mentions, and utterances of the character"""

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

class LabelIndependent(nn.Module):

    def __init__(self, pretrained_model_name, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        config = AutoConfig.from_pretrained(pretrained_model_name)
        hidden_size = config.hidden_size
        self.name_token_weights = nn.Linear(hidden_size, 1)
        self.name_weights = nn.Linear(hidden_size, 1)
        self.mention_token_weights = nn.Linear(hidden_size, 1)
        self.mention_weights = nn.Linear(hidden_size, 1)
        self.utterance_weights = nn.Linear(hidden_size, 1)
        self.combine_weights = nn.Linear(hidden_size, 1)

    def name_representation(self,
                            story_embeddings,
                            names_mask):
        """Create name representations

        Input
        =====
            story_embeddings = [n_blocks, sequence_length, hidden_size]
            names_mask = [n_characters, sequence_length]

        Output
        ======
            names_character_embeddings = [n_characters, hidden_size]
        """
        names_token_score = self.name_token_weights(story_embeddings).squeeze()
        # names_token_score = [n_blocks, seqlen]

        names_token_attn = torch.softmax(names_token_score.unsqueeze(dim=0) + names_mask.unsqueeze(dim=1), dim=2)
        # names_token_attn = [n_characters, n_blocks, seqlen]

        names_embeddings = torch.bmm(names_token_attn.permute(1, 0, 2), story_embeddings).permute(1, 0, 2)
        # names_embeddings = [n_characters, n_blocks, hidden_size]

        names_score = self.name_weights(names_embeddings).squeeze()
        # names_score = [n_characters, n_blocks]

        names_attn = torch.softmax(names_score, dim=1)
        # names_attn = [n_characters, n_blocks]

        names_character_embeddings = torch.bmm(names_attn.unsqueeze(dim=1), names_embeddings).squeeze()
        # names_character_embeddings = [n_characters, hidden_size]

        return names_character_embeddings

    def mention_representation(self,
                               n_characters,
                               story_embeddings,
                               mentions_mask,
                               mention_character_ids):
        """Create mention representations. Characters without mentions get zero-embeddings.

        Input
        =====
            n_characters = int
            story_embeddings = [n_blocks, sequence_length, hidden_size]
            mentions_mask = [n_mentions, total_sequence_length]
            mention_character_ids = [n_mentions]

        Output
        ======
            mentions_character_embeddings = [n_characters, hidden_size]
        """
        n_blocks, seqlen, hidden_size = story_embeddings.shape
        total_seqlen = n_blocks * seqlen
        story_embeddings = story_embeddings.reshape(total_seqlen, hidden_size)
        # story_embeddings = [total_seqlen, hidden_size]

        mentions_token_score = self.mention_token_weights(story_embeddings)
        # mentions_score = [total_seqlen, 1]

        mentions_token_attn = torch.softmax(mentions_mask + mentions_token_score.T, dim=1)
        # mentions_attn = [n_mentions, total_seqlen]

        mentions_embeddings = torch.mm(mentions_token_attn, story_embeddings)
        # mentions_embeddings = [n_mentions, hidden_size]

        mentions_score = self.mention_weights(mentions_embeddings)
        # mentions_score = [n_mentions, 1]

        device = next(self.parameters()).device
        # device where model is stored

        mentions_character_embeddings = torch.zeros((n_characters, hidden_size), device=device, dtype=float)
        # initialize mentions_embeddings = [n_characters, hidden_size]

        for i in mention_character_ids.unique():
            character_mention_mask = mention_character_ids == i
            # [n_mentions]

            character_mentions_embeddings = mentions_embeddings[character_mention_mask]
            # [n_character_mentions, hidden_size]

            character_mentions_score = mentions_score[character_mention_mask].T
            # [1, n_character_mentions]

            character_mentions_attn = torch.softmax(character_mentions_score, dim=1)
            # character_mentions_attn = [1, n_character_mentions]

            mentions_character_embeddings[i] = torch.mm(character_mentions_attn,
                                                        character_mentions_embeddings).squeeze()
            # mentions_character_embeddings[i] = [hidden_size]

        return mentions_character_embeddings

    def utterance_representation(self,
                                 story_embeddings,
                                 utterances_mask):
        """Create utterance representations. Characters without utterances get zero-embeddings.

        Input
        =====
            story_embeddings = [n_blocks, sequence_length, hidden_size]
            utterances_mask = [n_characters, total_sequence_length]

        Output
        ======
            utterance_embeddins = [n_characters, hidden_size]
        """
        n_blocks, seqlen, hidden_size = story_embeddings.shape
        total_seqlen = n_blocks * seqlen
        story_embeddings = story_embeddings.reshape(total_seqlen, hidden_size)
        # story_embeddings = [total_seqlen, hidden_size]

        utterance_score = self.utterance_weights(story_embeddings).T
        # utterance_score = [1, total_seqlen]

        utterance_attn = torch.softmax(utterances_mask + utterance_score, dim=1).nan_to_num(0)
        # utterance_attn = [n_characters, total_seqlen]

        utterance_embeddings = torch.mm(utterance_attn, story_embeddings)
        # utterance_embeddings = [n_characters, hidden_size]

        return utterance_embeddings

    def combine_embeddings(self,
                           name_embeddings,
                           mention_embeddings,
                           utterance_embeddings,
                           mention_character_ids,
                           utterances_mask):
        """Combine name, mention, and utterance embeddings

        Input
        =====
            name_embeddings = [n_characters, hidden_size]
            mention_embeddings = [n_characters, hidden_size]
            utterance_embeddings = [n_characters, hidden_size]
            mention_character_ids = [n_mentions]
            utterances_mask = [n_characters, total_sequence_length]

        Output
        ======
            character_embeddings = [n_characters, hidden_size]
        """
        device = next(self.parameters()).device
        n_characters, hidden_size = name_embeddings.shape
        embeddings = (torch.cat([name_embeddings, mention_embeddings, utterance_embeddings], dim=1)
                      .reshape(n_characters, 3, hidden_size))
        # embeddings = [n_characters, 3, hidden_size]

        score = self.combine_weights(embeddings).squeeze()
        # score = [n_characters, 3]

        mask = torch.zeros((n_characters, 3), device=device)
        mask[:, 0] = 1
        mask[mention_character_ids.unique(), 1] = 1
        mask[(utterances_mask == 0).any(dim=1), 2] = 1
        mask = torch.log(mask)
        # mask = [n_characters, 3]

        attn = torch.softmax(score + mask, dim=1)
        # attn = [n_characters, 3]

        character_embeddings = torch.bmm(attn.unsqueeze(dim=1), embeddings).squeeze()
        # character_embeddings = [n_characters, hidden_size]

        return character_embeddings

    def forward(self,
                story_input_ids,
                names_mask,
                utterances_mask,
                mentions_mask,
                mention_character_ids):
        """Create character representations

        Input
        =====
            story_input_ids = [n_blocks, sequence_length]
            names_mask = [n_characters, sequence_length]
            utterances_mask = [n_characters, total_sequence_length]
            mentions_mask = [n_mentions, total_sequence_length]
            mention_character_ids = [n_mentions]
            
            Here, total_sequence_length = n_blocks x max_sequence_length

        Output
        ======
            character_embeddings = [n_characters, hidden_size]
        """
        story_embeddings = self.encoder(story_input_ids).last_hidden_state
        # story_embeddings = [n_blocks, seqlen, hidden_size]

        name_embeddings = self.name_representation(story_embeddings, names_mask)
        # name_embeddins = [n_characters, hidden_size]

        n_characters = len(names_mask)
        mention_embeddins = self.mention_representation(n_characters, story_embeddings, mentions_mask, 
                                                        mention_character_ids)
        # mention_embeddings = [n_characters, hidden_size]

        utterance_embeddings = self.utterance_representation(story_embeddings, utterances_mask)
        # utterance_embeddings = [n_characters, hidden_size]

        character_embeddings = self.combine_embeddings(name_embeddings, mention_embeddins, utterance_embeddings)
        # character_embeddings = [n_characters, hidden_size]

        return character_embeddings