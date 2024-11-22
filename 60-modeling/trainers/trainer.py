"""Train the label-dependent or label-independent character representation models"""

from models.label_dependent import LabelDependent
from models.label_independent import LabelIndependent
from models.classifier import PortayClassifier
from models.pretrained import Model
from trainers import story_label_dependent_trainer
import datadirs

import os
import tqdm
import itertools
import pandas as pd
from absl import logging

import torch
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType

class Trainer:

    def __init__(self, args, expdir):
        self.args = args
        self.expdir = expdir

    def __call__(self):
        datadir = datadirs.datadir
        dataset_file = os.path.join(datadir, "CHATTER/chatter.csv")
        tropes_file = os.path.join(datadir, "CHATTER/tropes.csv")
        character_movie_map_file = os.path.join(datadir, "CHATTER/character-movie-map.csv")
        dataset_df = pd.read_csv(dataset_file, index_col=None, dtype={"content-text": str})
        tropes_df = pd.read_csv(tropes_file, index_col=None)
        character_movie_map_df = pd.read_csv(character_movie_map_file, index_col=None, dtype=str)
        tensors_dir = os.path.join(datadir, "60-modeling/tensors", self.args.input, self.args.model)
        tropes_dir = os.path.join(datadir, "60-modeling/trope-embeddings")

        # load trope embeddings
        tropes = tropes_df["trope"].tolist()
        trope_embeddings_files = sorted(os.listdir(tropes_dir))
        trope_embeddings_arr = []
        for file in tqdm.tqdm(trope_embeddings_files, desc="load trope embeddings", unit="file"):
            trope_embeddings_arr.append(torch.load(os.path.join(tropes_dir, file), weights_only=True))
        trope_embeddings = torch.cat(trope_embeddings_arr, dim=0).cuda()
        logging.info(f"trope-embeddings = {tuple(trope_embeddings.shape)}")

        # initialize model
        logging.info("initializing model")
        if self.args.model == "roberta":
            encoder = Model.roberta()
            target_modules = ["query", "key", "value"]
        elif self.args.model == "longformer":
            encoder = Model.longformer(self.args.longformerattn)
            target_modules = ["query", "key", "value", "global_query", "global_key", "global_value"]
        hidden_size = encoder.config.hidden_size
        if self.args.lbl:
            model = LabelDependent(hidden_size)
            classifier = PortayClassifier(hidden_size)
        else:
            model = LabelIndependent(hidden_size)
            classifier = PortayClassifier(2 * hidden_size)

        # lora
        logging.info("initializing lora adapters")
        loraconfig = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                                inference_mode=False,
                                r=self.args.rank,
                                lora_alpha=self.args.alpha,
                                target_modules=target_modules,
                                use_rslora=True,
                                lora_dropout=self.args.dropout,
                                bias="none")
        encoder = get_peft_model(encoder, loraconfig)

        # move model to gpu
        logging.info("moving model to gpu")
        encoder.cuda()
        model.cuda()
        classifier.cuda()

        # initialize optimizer
        logging.info("initializing optimizer")
        parameters = [{"params": itertools.chain(model.parameters(), classifier.parameters()), "lr": self.args.lr},
                      {"params": encoder.parameters(), "lr": self.args.elr}]
        optimizer = AdamW(parameters, weight_decay=self.args.decay)

        # train model
        if self.args.input == "story":
            logging.info("input = story")
            if self.args.lbl:
                logging.info("model = label-dependent\n")
                story_label_dependent_trainer.train(encoder,
                                                    model,
                                                    classifier,
                                                    optimizer,
                                                    dataset_df,
                                                    character_movie_map_df,
                                                    tropes,
                                                    trope_embeddings,
                                                    tensors_dir,
                                                    self.args.ep,
                                                    1000 * self.args.tokbatch,
                                                    self.args.trpbatch,
                                                    self.args.gradnorm,
                                                    self.args.batchepoch,
                                                    self.args.batcheval)