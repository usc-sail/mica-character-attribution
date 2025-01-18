"""Base class for creating character representations conditioned on label representations"""
import torch
from torch import nn

class LabelDependent(nn.Module):
    """Base class for creating character representations conditioned on label representations"""

    def __init__(self, hidden_size: int, use_mentions: bool, use_utterances: bool) -> None:
        super().__init__()
        self.use_mentions = use_mentions
        self.use_utterances = use_utterances
        self.name_weights = nn.Linear(hidden_size, 1)
        self.mention_weights = nn.Linear(hidden_size, 1)
        self.combine_weights = nn.Linear(hidden_size, 1)

    def combine_embeddings(self, name_embeddings, *embeddings_tup):
        """Combine name representation with only mention, only utterance, or both mention and utterance representations

        Input
        -----
            name_embeddings: [n_characters, n_labels, hidden_size]
            embeddings_tup: tuple of [n_characters, n_labels, hidden_size]

        Output
        ------
            character_embeddings: [n_characters, n_labels, hidden_size]
        """
        device = next(self.parameters()).device
        n_characters, n_labels, hidden_size = name_embeddings.shape
        embeddings_arr = [name_embeddings] + list(embeddings_tup) # m-length list of [k, t, d]
        embeddings = torch.cat(embeddings_arr, dim=2) # [k, t, md]
        embeddings = embeddings.reshape(n_characters, n_labels, -1, hidden_size) # [k, t, m, d]
        score = self.combine_weights(embeddings).squeeze(dim=3) # [k, t, m]
        mask_arr = [torch.any(embeddings != 0, dim=2) for embeddings in embeddings_tup] # m-1 length list of [k, t]
        full_mask = torch.full((n_characters, n_labels), fill_value=True, dtype=bool, device=device) # [k, t]
        mask_arr = [full_mask] + mask_arr # m-length list of [k, t]
        mask = torch.cat(mask_arr, dim=1) # [k, mt]
        mask = mask.reshape(n_characters, -1, n_labels) # [k, m, t]
        mask = mask.permute(0, 2, 1) # [k, t, m]
        mask = torch.log(mask) # [k, t, m]
        attn = torch.softmax(score + mask, dim=2) # [k, t, m]
        attn = attn.unsqueeze(dim=2) # [k, t, 1, m]
        combined_embeddings = torch.matmul(attn, embeddings).squeeze(dim=2) # [k, t, d]
        return combined_embeddings