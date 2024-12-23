"""Base class for creating character representations independent of label representations"""
import torch
from torch import nn

class LabelIndependent(nn.Module):
    """Base class for creating character representations independent of label representations"""

    def __init__(self, hidden_size, use_mentions, use_utterances) -> None:
        super().__init__()
        self.use_mentions = use_mentions
        self.use_utterances = use_utterances
        self.name_weights = nn.Linear(hidden_size, 1)
        self.name_seq_weights = nn.Linear(hidden_size, 1)
        self.mention_weights = nn.Linear(hidden_size, 1)
        self.mention_seq_weights = nn.Linear(hidden_size, 1)
        self.utterance_seq_weights = nn.Linear(hidden_size, 1)
        self.combine_weights = nn.Linear(hidden_size, 1)

    def combine_embeddings(self,
                           name_embeddings,
                           *embeddings_tup):
        """Combine name representation with only mention, only utterance, or both mention and utterance representations

        Input
        -----
            name_embeddings: [n_characters, hidden_size]
            embeddings_tup: tuple of [n_characters, hidden_size]

        Output
        ------
            character_embeddings: [n_characters, hidden_size]
        """
        device = next(self.parameters()).device
        n_characters, hidden_size = name_embeddings.shape
        embeddings_arr = [name_embeddings] + list(embeddings_tup) # m-length list of [k, d]
        embeddings = torch.cat(embeddings_arr, dim=1) # [k, md]
        embeddings = embeddings.reshape(n_characters, -1, hidden_size) # [k, m, d]
        score = self.combine_weights(embeddings).squeeze(dim=2) # [k, m]
        mask_arr = [torch.any(embeddings != 0, dim=1) for embeddings in embeddings_tup] # m-1 length list of [k]
        full_mask = torch.full((n_characters,), fill_value=True, dtype=bool, device=device) # [k]
        mask_arr = [full_mask] + mask_arr # m-length list of [k]
        mask = torch.cat(mask_arr, dim=0) # [mk]
        mask = mask.reshape(-1, n_characters) # [m, k]
        mask = mask.T # [k, m]
        mask = torch.log(mask) # [k, m]
        attn = torch.softmax(score + mask, dim=1) # [k, m]
        attn = attn.unsqueeze(dim=1) # [k, 1, m]
        combined_embeddings = torch.matmul(attn, embeddings).squeeze(dim=1) # [k, d]
        return combined_embeddings