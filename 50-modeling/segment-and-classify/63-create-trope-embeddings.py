import datadirs
from models import pretrained

from absl import app
from absl import flags
import math
import os
import pandas as pd
import shutil
import torch
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_enum("model", default="roberta", enum_values=["roberta", "longformer", "llama"], help="model")
flags.DEFINE_integer("batch", default=192, help="number of tropes per batch")
# 192 works good for longformer, 300 for roberta in a 48 GB GPU

def create_trope_embeddings(_):
    model = FLAGS.model
    batchsize = FLAGS.batch
    datadir = datadirs.datadir
    tropes_file = os.path.join(datadir, "CHATTER/tropes.csv")
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    if model == "roberta":
        tokenizer = pretrained.Tokenizer.roberta()
        encoder = pretrained.Model.roberta()
    elif model == "longformer":
        tokenizer = pretrained.Tokenizer.longformer()
        encoder = pretrained.Model.longformer(512)
    elif model == "llama":
        tokenizer = pretrained.Tokenizer.llama()
        encoder = pretrained.Model.llama()
    encoder.cuda()
    definitions = tropes_df["summary"].tolist()
    tokenizer_output = tokenizer(definitions, padding="max_length", truncation=True, max_length=128,
                                 return_tensors="pt", return_attention_mask=True)
    input_ids = tokenizer_output.input_ids.cuda()
    attention_mask = tokenizer_output.attention_mask.cuda()
    nbatches = math.ceil(len(definitions)/batchsize)
    tensorsdir = os.path.join(datadir, f"60-modeling/tensors/tropes/{model}")
    trope_names_file = os.path.join(datadir, "60-modeling/tensors/tropes/tropes.txt")
    os.makedirs(tensorsdir, exist_ok=True)
    with open(trope_names_file, "w") as fw:
        fw.write("\n".join(tropes_df["trope"].tolist()))
    for i in tqdm.trange(nbatches, unit="batch", desc="encoding tropes"):
        batch_input_ids = input_ids[i * batchsize : (i + 1) * batchsize]
        batch_attention_mask = attention_mask[i * batchsize : (i + 1) * batchsize]
        batch_encoder_output = encoder(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        batch_trope_embeddings = torch.mean(batch_encoder_output.last_hidden_state, dim=1)
        strfileid = f"{i + 1}".zfill(3)
        batch_trope_embeddings_file = os.path.join(tensorsdir, f"trope-embeddings-{strfileid}.pt")
        torch.save(batch_trope_embeddings, batch_trope_embeddings_file)
    trope_embeddings_arr = []
    trope_embeddings_file = os.path.join(datadir, f"60-modeling/tensors/tropes/{model}.pt")
    for i in tqdm.trange(nbatches, unit="file", desc="concatenation"):
        strfileid = f"{i + 1}".zfill(3)
        batch_trope_embeddings_file = os.path.join(tensorsdir, f"trope-embeddings-{strfileid}.pt")
        trope_embeddings_arr.append(torch.load(batch_trope_embeddings_file, weights_only=True))
    trope_embeddings = torch.cat(trope_embeddings_arr, dim=0)
    print(f"embeddings shape = {tuple(trope_embeddings.shape)} dtype = {trope_embeddings.dtype}")
    torch.save(trope_embeddings, trope_embeddings_file)
    shutil.rmtree(tensorsdir)

if __name__ == '__main__':
    app.run(create_trope_embeddings)