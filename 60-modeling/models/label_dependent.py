"""Find character representations from the name, mentions, and utterances of the character conditioned on the labels"""

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

class LabelDependent(nn.Module):

    def __init__(self, pretrained_model_name, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        config = AutoConfig.from_pretrained(pretrained_model_name)
        self.hidden_size = config.hidden_size
        self.name_weights = nn.Linear(self.hidden_size, 1)
        self.mentions_weights = nn.Linear(self.hidden_size, 1)
        self.combine_weights = nn.Linear(self.hidden_size, 1)

    @property
    def encoder_parameters(self):
        return self.encoder.parameters()

    @property
    def non_encoder_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("encoder"):
                yield parameter

    def name_representation(self,
                            story_embeddings,
                            label_embeddings,
                            names_mask):
        """Create label-dependent name representations

        Input
        =====
            story_embeddings = [n_blocks, sequence_length, hidden_size]
            label_embeddings = [n_labels, hidden_size]
            names_mask = [n_characters, sequence_length]

        Output
        ======
            name_label_embeddings = [n_characters, n_labels, hidden_size]
        """
        names_score = self.name_weights(story_embeddings).squeeze(dim=2)
        # names_score = [n_blocks, seqlen]

        names_attn = torch.softmax(names_score.unsqueeze(dim=0) + names_mask.unsqueeze(dim=1), dim=2)
        # names_attn = [n_characters, n_blocks, seqlen]

        names_embeddings = torch.bmm(names_attn.permute(1, 0, 2), story_embeddings).permute(1, 0, 2)
        # names_embeddings = [n_characters, n_blocks, hidden_size]

        name_label_score = (torch.matmul(names_embeddings, label_embeddings.unsqueeze(dim=0).permute(0, 2, 1))
                            .permute(0, 2, 1))
        # name_label_score = [n_characters, n_labels, n_blocks]

        name_label_attn = torch.softmax(name_label_score, dim=2)
        # name_label_attn = [n_characters, n_labels, n_blocks]

        name_label_embeddings = torch.bmm(name_label_attn, names_embeddings)
        # name_label_embeddings = [n_characters, n_labels, hidden_size]

        return name_label_embeddings

    def mention_representation(self,
                               n_characters,
                               story_embeddings,
                               label_embeddings,
                               mentions_mask,
                               mention_character_ids):
        """Create label-dependent mention representations. Characters without mentions get zero-embeddings.

        Input
        =====
            n_characters = int
            story_embeddings = [n_blocks, sequence_length, hidden_size]
            label_embeddins = [n_labels, hidden_size]
            mentions_mask = [n_mentions, total_sequence_length]
            mention_character_ids = [n_mentions]

        Output
        ======
            mention_label_embeddings = [n_characters, n_labels, hidden_size]
        """
        n_labels = len(label_embeddings)
        n_blocks, seqlen, hidden_size = story_embeddings.shape
        total_seqlen = n_blocks * seqlen
        story_embeddings = story_embeddings.reshape(total_seqlen, hidden_size)
        # story_embeddings = [total_seqlen, hidden_size]

        mentions_score = self.mentions_weights(story_embeddings)
        # mentions_score = [total_seqlen, 1]

        mentions_attn = torch.softmax(mentions_mask + mentions_score.T, dim=1)
        # mentions_attn = [n_mentions, total_seqlen]

        mentions_embeddings = torch.mm(mentions_attn, story_embeddings)
        # mentions_embeddings = [n_mentions, hidden_size]

        mention_label_score = torch.mm(mentions_embeddings, label_embeddings.T)
        # mention_label_score = [n_mentions, n_labels]

        device = next(self.parameters()).device
        # device where model is stored

        mention_label_embeddings = torch.zeros((n_characters, n_labels, hidden_size), device=device,
                                               dtype=torch.float32)
        # initialize mention_label_embeddings = [n_characters, n_labels, hidden_size]

        for i in mention_character_ids.unique():
            character_mention_mask = mention_character_ids == i
            # [n_mentions]

            character_mentions_embeddings = mentions_embeddings[character_mention_mask]
            # [n_character_mentions, hidden_size]

            character_mention_label_score = mention_label_score[character_mention_mask].T
            # [n_labels, n_character_mentions]

            character_mention_label_attn = character_mention_label_score.softmax(dim=1)
            # character_mention_label_attn = [n_labels, n_character_mentions]

            mention_label_embeddings[i] = torch.mm(character_mention_label_attn, character_mentions_embeddings)
            # mention_label_embeddings[i] = [n_labels, hidden_size]

        return mention_label_embeddings

    def utterance_representation(self,
                                 story_embeddings,
                                 label_embeddings,
                                 utterances_mask):
        """Create label-dependent utterance representations. Characters without utterances get zero-embeddings.

        Input
        =====
            story_embeddings = [n_blocks, sequence_length, hidden_size]
            label_embeddings = [n_labels, hidden_size]
            utterances_mask = [n_characters, total_sequence_length]

        Output
        ======
            utterance_embeddins = [n_characters, n_labels, hidden_size]
        """
        n_blocks, seqlen, hidden_size = story_embeddings.shape
        total_seqlen = n_blocks * seqlen
        story_embeddings = story_embeddings.reshape(total_seqlen, hidden_size)
        # story_embeddings = [total_seqlen, hidden_size]

        score = torch.mm(label_embeddings, story_embeddings.T)
        # score = [n_labels, total_seqlen]

        attn = torch.softmax(utterances_mask.unsqueeze(dim=1) + score.unsqueeze(dim=0), dim=2).nan_to_num(0)
        # attn = [n_characters, n_labels, total_seqlen]

        utterance_embeddings = torch.matmul(attn, story_embeddings.unsqueeze(dim=0))
        # utterance_embeddings = [n_characters, n_labels, hidden_size]

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
            name_embeddings = [n_characters, n_labels, hidden_size]
            mention_embeddings = [n_characters, n_labels, hidden_size]
            utterance_embeddings = [n_characters, n_labels, hidden_size]
            mention_character_ids = [n_mentions]
            utterances_mask = [n_characters, total_sequence_length]

        Output
        ======
            character_embeddings = [n_characters, n_labels, hidden_size]
        """
        device = next(self.parameters()).device
        n_characters, n_labels, hidden_size = name_embeddings.shape
        embeddings = (torch.cat([name_embeddings, mention_embeddings, utterance_embeddings], dim=2)
                      .reshape(n_characters, n_labels, 3, hidden_size))
        # embeddings = [n_characters, n_labels, 3, hidden_size]

        score = self.combine_weights(embeddings).squeeze(dim=3)
        # score = [n_characters, n_labels, 3]

        mask = torch.zeros((n_characters, 3), device=device)
        mask[:, 0] = 1
        mask[mention_character_ids.unique(), 1] = 1
        mask[(utterances_mask == 0).any(dim=1), 2] = 1
        mask = torch.log(mask)
        # mask = [n_characters, 3]

        attn = torch.softmax(score + mask.unsqueeze(dim=1), dim=2)
        # attn = [n_characters, n_labels, 3]

        character_embeddings = torch.matmul(attn.unsqueeze(dim=2), embeddings).squeeze(dim=2)
        # character_embeddings = [n_characters, n_labels, hidden_size]

        return character_embeddings

    def forward(self,
                story_input_ids,
                names_mask,
                utterances_mask,
                mentions_mask,
                mention_character_ids,
                label_input_ids):
        """Create character representations conditioned on the labels

        Input
        =====
            story_input_ids = [n_blocks, sequence_length]
            names_mask = [n_characters, sequence_length]
            utterances_mask = [n_characters, total_sequence_length]
            mentions_mask = [n_mentions, total_sequence_length]
            mention_character_ids = [n_mentions]
            label_input_ids = [n_labels, max_sequence_length]
            
            Here, total_sequence_length = n_blocks x max_sequence_length

        Output
        ======
            character_embeddings = [n_characters, n_labels, hidden_size]
        """
        story_embeddings = self.encoder(story_input_ids).last_hidden_state
        # story_embeddings = [n_blocks, seqlen, hidden_size]

        label_embeddings = self.encoder(label_input_ids).pooler_output
        # label_embeddings = [n_labels, hidden_size]

        name_label_embeddings = self.name_representation(story_embeddings, label_embeddings, names_mask)
        # name_label_embeddins = [n_characters, n_labels, hidden_size]

        n_characters = len(names_mask)
        mention_label_embeddins = self.mention_representation(n_characters, story_embeddings, label_embeddings,
                                                              mentions_mask, mention_character_ids)
        # mention_label_embeddings = [n_characters, n_labels, hidden_size]

        utterance_label_embeddings = self.utterance_representation(story_embeddings, label_embeddings, utterances_mask)
        # utterance_label_embeddings = [n_characters, n_labels, hidden_size]

        character_embeddings = self.combine_embeddings(name_label_embeddings, mention_label_embeddins,
                                                       utterance_label_embeddings, mention_character_ids,
                                                       utterances_mask)
        # character_embeddings = [n_characters, n_labels, hidden_size]

        return character_embeddings