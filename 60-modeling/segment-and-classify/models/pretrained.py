import torch
import transformers
from transformers import AutoModel, AutoConfig, AutoTokenizer

class Tokenizer:

    def roberta():
        return AutoTokenizer.from_pretrained("roberta-base", use_fast=True, add_prefix_space=True)

    def longformer():
        config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
        tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=True, add_prefix_space=True)
        tokenizer.model_max_length = config.max_position_embeddings-2
        return tokenizer

    def llama():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True,
                                                  add_prefix_space=True)
        tokenizer.model_max_length = 1 << 14
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        return tokenizer

class Model:

    def roberta():
        return AutoModel.from_pretrained("roberta-base", add_pooling_layer=False)

    def longformer(attnwindow: int):
        config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
        config.attention_window = attnwindow
        return AutoModel.from_pretrained("allenai/longformer-base-4096", config=config, add_pooling_layer=False)

    def llama():
        tokenizer = Tokenizer.llama()
        model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        return model