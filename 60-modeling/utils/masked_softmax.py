import torch

def masked_softmax(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Perform softmax on last dimension of `score` matrix.

    The `mask` matrix is a 0/-inf matrix.
    `score` and `mask` should be broadcast-compatible.
    `mask` can mask out complete rows.
    Those rows would get 0 value in the output.
    """
    attn = score + mask
    fully_masked = attn.isinf().all(dim=-1)
    attn[~fully_masked] = torch.softmax(attn[~fully_masked], dim=-1)
    attn[fully_masked] = 0
    return attn