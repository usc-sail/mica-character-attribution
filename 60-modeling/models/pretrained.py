from transformers import AutoModel, AutoConfig, AutoTokenizer

class Tokenizer:

    def roberta():
        return AutoTokenizer.from_pretrained("roberta-base", use_fast=True, add_prefix_space=True)

    def longformer():
        config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
        tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=True, add_prefix_space=True)
        tokenizer.model_max_length = config.max_position_embeddings
        return tokenizer

class Model:

    def roberta():
        return AutoModel.from_pretrained("roberta-base")

    def longformer():
        return AutoModel.from_pretrained("allenai/longformer-base-4096")