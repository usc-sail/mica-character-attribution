"""Model probability of a character to portray some trait from a character x trait embedding matrix or from
separate character and trait embedding matrices
"""
from torch import nn

class PortayClassifier(nn.Module):

    def __init__(self, input_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ffn = nn.Linear(input_size, 1)

    def forward(self, mat):
        return self.ffn(mat).squeeze(dim=-1)