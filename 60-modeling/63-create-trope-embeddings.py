import datadirs

import os
import math
import tqdm
import torch
import pandas as pd
from transformers import RobertaModel, RobertaTokenizer

def create_trope_embeddings():
    datadir = datadirs.datadir
    tropes_file = os.path.join(datadir, "CHATTER/tropes.csv")
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    encoder = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)
    encoder.cuda()
    definitions = tropes_df["summary"].tolist()
    tokenizer_output = tokenizer(definitions, padding="max_length", truncation=True, max_length=128,
                                 return_tensors="pt", return_attention_mask=True)
    input_ids = tokenizer_output.input_ids.cuda()
    attention_mask = tokenizer_output.attention_mask.cuda()
    batchsize = 300
    nbatches = math.ceil(len(definitions)/batchsize)
    for i in tqdm.trange(nbatches):
        batch_input_ids = input_ids[i * batchsize : (i + 1) * batchsize]
        batch_attention_mask = attention_mask[i * batchsize : (i + 1) * batchsize]
        batch_encoder_output = encoder(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        batch_trope_embeddings = torch.mean(batch_encoder_output.last_hidden_state, dim=1)
        strfileid = f"{i + 1}".zfill(3)
        batch_trope_embeddings_file = os.path.join(datadir,
                                                   f"60-modeling/trope-embeddings/trope-embeddings-{strfileid}.pt")
        torch.save(batch_trope_embeddings, batch_trope_embeddings_file)

if __name__ == '__main__':
    create_trope_embeddings()