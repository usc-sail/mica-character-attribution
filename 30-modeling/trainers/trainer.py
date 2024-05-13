"""Train the label-dependent or label-independent character representation models"""

from ..models.label_dependent import LabelDependent
from ..models.label_independent import LabelIndependent

import os
import pandas as pd
from typing import Any
from bs4 import BeautifulSoup as bs
from transformers import AutoTokenizer

class Trainer:

    def __init__(self,
                 data_dir,
                 splits_file,
                 tropes_file,
                 data_type,
                 pretrained_model_name,
                 tokenizer_model_name,
                 label_dependent
                 ) -> None:
        self.data_dir = data_dir
        self.splits_file = splits_file
        self.tropes_file = tropes_file
        self.data_type = data_type
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer_model_name = tokenizer_model_name
        self.label_dependent = label_dependent

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        splits_file = os.path.join(self.data_dir, self.splits_file)
        tropes_file = os.path.join(self.data_dir, self.tropes_file)
        splits_df = pd.read_csv(splits_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
        tropes_df = pd.read_csv(tropes_file, index_col="trope")
        tensors_dir = os.path.join(self.data_dir, "30-modeling/tensors", self.data_type, self.tokenizer_model_name)

        # initialize model
        if self.label_dependent:
            self.model = LabelDependent(self.pretrained_model_name).cuda()
        else:
            self.model = LabelIndependent(self.pretrained_model_name).cuda()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=True, add_prefix_space=True)

        # encode traits
        tropes = splits_df["trope"].unique()
        definitions = [bs(tropes_df.loc[trope, "definition"]).text for trope in tropes]
        definition_token_ids = (tokenizer(definitions, padding="max_length", truncation=True, return_tensors="pt")
                                .input_ids).cuda()
        definition_embeddings = self.model.encoder(definition_token_ids).pooler_output

        # read tensors
        if self.data_type == "story":
            imdb_ids = splits_df["imdb-id"].unique()
            