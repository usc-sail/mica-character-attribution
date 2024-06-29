"""Train the label-dependent or label-independent character representation models"""

from models.label_dependent import LabelDependent
from models.label_independent import LabelIndependent
from models.classifier import SingleRepresentationClassifier, CompareRepresentationClassifier
from trainers import story_label_dependent_trainer

import os
import tqdm
import torch
import random
import itertools
import numpy as np
import pandas as pd
from typing import Any
from torch.optim import AdamW
from bs4 import BeautifulSoup as bs
from transformers import AutoTokenizer

class Trainer:

    def __init__(self,
                 data_dir,
                 dataset_file,
                 tropes_file,
                 data_type,
                 pretrained_model_name,
                 tokenizer_model_name,
                 label_dependent,
                 n_epochs,
                 encoderlr,
                 lr,
                 freeze_encoder
                 ) -> None:
        self.data_dir = data_dir
        self.dataset_file = dataset_file
        self.tropes_file = tropes_file
        self.data_type = data_type
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer_model_name = tokenizer_model_name
        self.label_dependent = label_dependent
        self.n_epochs = n_epochs
        self.encoderlr = encoderlr
        self.lr = lr
        self.freeze_encoder = freeze_encoder

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        dataset_file = os.path.join(self.data_dir, self.dataset_file)
        tropes_file = os.path.join(self.data_dir, self.tropes_file)
        dataset_df = pd.read_csv(dataset_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
        tropes_df = pd.read_csv(tropes_file, index_col="trope")
        tensors_dir = os.path.join(self.data_dir, "60-modeling/tensors", self.data_type, self.tokenizer_model_name)
        train_df = dataset_df[dataset_df["partition"] == "train"]
        train_components = np.random.choice(train_df["component"].unique(), 10)
        train_df = train_df[train_df["component"].isin(train_components)]

        # initialize model
        if self.label_dependent:
            model = LabelDependent(self.pretrained_model_name)
            classifier = SingleRepresentationClassifier(model.hidden_size)
        else:
            model = LabelIndependent(self.pretrained_model_name)
            classifier = CompareRepresentationClassifier(model.hidden_size)

        # move model to gpu
        model.cuda()
        classifier.cuda()

        # freeze encoder if flag is set
        if self.freeze_encoder:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False

        # initialize optimizer
        parameters = [{"params": model.non_encoder_parameters}, {"params": classifier.parameters()}]
        if not self.freeze_encoder:
            parameters.append({"params": model.encoder_parameters, "lr": self.encoderlr})
        optimizer = AdamW(parameters, lr=self.lr)

        # encode traits
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=True, add_prefix_space=True)
        tropes = sorted(dataset_df["trope"].unique())
        definitions = [tropes_df.loc[trope, "definition"] for trope in tropes]
        trope_token_ids = (tokenizer(definitions, padding="max_length", truncation=True, return_tensors="pt")
                           .input_ids).cuda()

        # train model
        if self.label_dependent and self.data_type == "story":
            story_label_dependent_trainer.train(model, classifier, optimizer, train_df, tropes, trope_token_ids,
                                                tensors_dir, self.n_epochs)