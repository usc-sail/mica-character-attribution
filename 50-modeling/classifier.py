# Classification model for the character attribution task
from accelerate import PartialState
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification

PARTIALSTATE = PartialState()

class BinaryClassifier(nn.Module):
    """Takes input as '[ATTRIBUTE] <Attribute> [CHARACTER] <Character> [CONTEXT] <Context>', applies a sequence
    classification model, polls the CLS representation, and feeds it to FFN"""

    def __init__(self, modelname, compute_dtype, quantization_config, attn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder =  AutoModelForSequenceClassification.from_pretrained(modelname,
                                                                           num_labels=2,
                                                                           torch_dtype=compute_dtype,
                                                                           quantization_config=quantization_config,
                                                                           device_map={"": PARTIALSTATE.process_index},
                                                                           attn_implementation=attn)
        self.encoder.config.pad_token_id = self.encoder.config.eos_token_id

    def forward(self, batch):
        return self.encoder(**batch)

class BinarySiameseClassifier(nn.Module):
    """Takes as input '[CHARACTER] <Character> [CONTEXT] <Context>' and attribute definitions, apply a language
    model to get character and attribute representations, apply a siamese network to get distance, use cross-entropy or ranking loss"""

class MulticlassClassifier(nn.Module):
    """Takes as input '[ATTRIBUTE_1] <Attribute_1> [ATTRIBUTE_2] <Attribute_2> ... [ATTRIBUTE_m] <Attribute_m>
    [CHARACTER] <Character> [CONTEXT] <Context>', applies a language model, polls the representations corresponding
    to the [ATTRIBUTE_n] tokens, and feed it to FFN"""

    def __init__(self, modelname, compute_dtype, quantization_config, attn, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = AutoModel.from_pretrained(modelname,
                                                 torch_dtype=compute_dtype,
                                                 quantization_config=quantization_config,
                                                 device_map={"": PARTIALSTATE.process_index},
                                                 attn_implementation=attn)
        self.score = nn.Linear(self.encoder.config.hidden_size, n_classes)

    def forward(self, batch):
        out = self.encoder(**batch)

class MulticlassSiameseClassifier(nn.Module):
    """Takes as input '[CHARACTER] <Character> [CONTEXT] <Context>' and attribute definitions, applies a language 
    model to get character representation and attribute representations, apply a siamese network to get distances
    between character and attribute representation, use cross-entropy or ranking loss"""