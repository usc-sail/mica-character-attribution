# Classification model for the character attribution task
import data

from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, modeling_outputs

class BinaryClassifier(AutoModelForSequenceClassification):
    """Takes input as '[ATTRIBUTE] <Attribute> [CHARACTER] <Character> [CONTEXT] <Context>', applies a sequence
    classification model, polls the CLS representation, and feeds it to FFN"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

class BinarySiameseClassifier(AutoModel):
    """Takes as input '[CHARACTER] <Character> [CONTEXT] <Context>' and attribute definitions, apply a language
    model to get character and attribute representations, apply a siamese network to get distance, use cross-entropy or ranking loss"""

class MulticlassClassifier(AutoModel):
    """Takes as input '[ATTRIBUTE_1] <Attribute_1> [ATTRIBUTE_2] <Attribute_2> ... [ATTRIBUTE_m] <Attribute_m>
    [CHARACTER] <Character> [CONTEXT] <Context>', applies a language model, polls the representations corresponding
    to the [ATTRIBUTE_n] tokens, and feed it to FFN"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score = nn.Linear(self.config.hidden_size, 1, device=next(super().parameters()).device)
        self.attribute_token_id = None

    def forward(self, input_ids=None, labels=None, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        attribute = out.last_hidden_state[input_ids == self.attribute_token_id].reshape(
            -1, data.NCLASSES, self.config.hidden_size)
        score = self.score(attribute).squeeze()
        loss = nn.functional.cross_entropy(score, labels)
        return modeling_outputs.SequenceClassifierOutput(loss=loss, logits=score)

