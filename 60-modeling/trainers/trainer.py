"""Train the label-dependent or label-independent character representation models"""

from models.label_dependent import LabelDependent
from models.label_independent import LabelIndependent
from models.classifier import PortayClassifier
from trainers import story_label_dependent_trainer, character_label_dependent_trainer
from trainers import story_label_independent_trainer, character_label_independent_trainer

import os
import pandas as pd
from torch.optim import AdamW
from transformers import AutoTokenizer

class Trainer:

    def __init__(self, args):
        self.args = args

    def __call__(self):
        dataset_file = os.path.join(self.args.datadir, "CHATTER/chatter.csv")
        tropes_file = os.path.join(self.args.datadir, "CHATTER/tropes.csv")
        character_movie_map_file = os.path.join(self.args.datadir, "CHATTER/character-movie-map.csv")
        dataset_df = pd.read_csv(dataset_file, index_col=None, dtype={"content-text": str})
        tropes_df = pd.read_csv(tropes_file, index_col="trope")
        character_movie_map_df = pd.read_csv(character_movie_map_file, index_col=None, dtype=str)
        tensors_dir = os.path.join(self.args.datadir, "60-modeling/tensors", self.args.inputtype,
                                   self.args.tokenizermodel)

        # initialize model
        print("initializing model")
        if self.args.labeldependent:
            model = LabelDependent(self.args.pretrainedmodel)
            classifier = PortayClassifier(model.hidden_size)
        else:
            model = LabelIndependent(self.args.pretrainedmodel)
            classifier = PortayClassifier(2 * model.hidden_size)
        print("\n")

        # move model to gpu
        print("moving model to gpu")
        model.cuda()
        classifier.cuda()

        # freeze encoder if flag is set
        if self.args.freezeencoder:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False

        # initialize optimizer
        print("initializing optimizer")
        parameters = [{"params": model.non_encoder_parameters}, {"params": classifier.parameters()}]
        if not self.args.freezeencoder:
            parameters.append({"params": model.encoder_parameters, "lr": self.args.encoderlr})
        optimizer = AdamW(parameters, lr=self.args.lr)
        print("\n")

        # encode traits
        print("encoding trope definitions")
        tokenizer = AutoTokenizer.from_pretrained(self.args.pretrainedmodel, use_fast=True, add_prefix_space=True)
        tropes = sorted(dataset_df["trope"].unique())
        definitions = [tropes_df.loc[trope, "definition"] for trope in tropes]
        trope_token_ids = (tokenizer(definitions, padding="max_length", truncation=True, return_tensors="pt")
                           .input_ids).cuda()

        print("\n")

        # train model
        if self.args.inputtype == "story":
            print("input-type = story")
            if self.args.labeldependent:
                print("model-type = label-dependent\n\n")
                story_label_dependent_trainer.train(model, classifier, optimizer, dataset_df, character_movie_map_df,
                                                    tropes, trope_token_ids, tensors_dir, self.args.nepochs,
                                                    self.args.ntropesbatch)