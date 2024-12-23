import torch

def cartesian_concatenate(character_embeddings: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
    """Concatenate every pair of character and label embeddings

    Suppose `character_embeddings` is [k, d] where k is the number of characters.
    Suppose `label_embeddings` is [t, d] where t is the number of labels.
    d is the hidden size.
    Then, we want to create the matrix Z of shape [k, t, 2d].
    Z[i, j] is the concatenation of the ith character embedding and jth label embedding.
    """
    n_characters = character_embeddings.shape[0]
    n_labels = label_embeddings.shape[0]
    character_embeddings = character_embeddings.unsqueeze(dim=1) # [k, 1, d]
    label_embeddings = label_embeddings.unsqueeze(dim=0) # [1, t, d]
    character_embeddings = character_embeddings.expand(-1, n_labels, -1) # [k, t, d]
    label_embeddings = label_embeddings.expand(n_characters, -1, -1) # [k, t, d]
    embeddings = torch.cat([character_embeddings, label_embeddings], dim=2) # [k, t, 2d]
    return embeddings