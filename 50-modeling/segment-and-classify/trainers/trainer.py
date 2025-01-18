"""Train the label-dependent or label-independent character representation models"""

from models.labeldep.character import CharacterLabelDependent
from models.labeldep.story import StoryLabelDependent
from models.labelind.character import CharacterLabelIndependent
from models.labelind.story import StoryLabelIndependent
from models.classifier import PortayClassifier
from models.pretrained import Model
from trainers import character_trainer
from trainers import story_trainer
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
        inputtype = "character" if self.args.segment else "story"
        tensors_dir = os.path.join(datadir, "60-modeling/tensors", inputtype, self.args.model)
        if self.args.segment:
            tensors_dir = os.path.join(tensors_dir, f"context-{self.args.context}")
        trope_embeddings_file = os.path.join(datadir, f"60-modeling/tensors/tropes/{self.args.model}.pt")

        # load trope embeddings
        tropes = tropes_df["trope"].tolist()
        trope_embeddings = torch.load(trope_embeddings_file, weights_only=True).cuda()
        logging.info(f"trope-embeddings = {tuple(trope_embeddings.shape)}\n")

        # initialize model
        logging.info("initializing model")
        if self.args.model == "roberta":
            encoder = Model.roberta()
            target_modules = ["query", "key", "value"]
        elif self.args.model == "longformer":
            encoder = Model.longformer(self.args.longformerattn)
            target_modules = ["query", "key", "value", "global_query", "global_key", "global_value"]
        elif self.args.model == "llama":
            encoder = Model.llama()
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        hidden_size = encoder.config.hidden_size
        if self.args.lbl:
            if self.args.segment:
                model = CharacterLabelDependent(hidden_size, self.args.mention, self.args.utter)
            else:
                model = StoryLabelDependent(hidden_size, self.args.mention, self.args.utter)
            classifier = PortayClassifier(hidden_size)
        else:
            if self.args.segment:
                model = CharacterLabelIndependent(hidden_size, self.args.mention, self.args.utter)
            else:
                model = StoryLabelIndependent(hidden_size, self.args.mention, self.args.utter)
            classifier = PortayClassifier(2 * hidden_size)

        # lora
        if self.args.lora:
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
        logging.info("moving model to gpu\n")
        encoder.cuda()
        model.cuda()
        classifier.cuda()

        # initialize optimizer
        logging.info("initializing optimizer\n")
        parameters = [{"params": itertools.chain(model.parameters(), classifier.parameters()), "lr": self.args.lr},
                      {"params": encoder.parameters(), "lr": self.args.elr}]
        optimizer = AdamW(parameters, weight_decay=self.args.decay)

        # train model
        if self.args.segment:
            character_trainer.train(self.args.lbl,
                                    encoder,
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
        else:
            story_trainer.train(self.args.lbl,
                                encoder,
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