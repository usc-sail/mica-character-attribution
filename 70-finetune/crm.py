"""Training and prediction functions for character representations modeling"""
import data
import utils

from absl import flags
from absl import logging
from accelerate import PartialState
import math
import numpy as np
import os
import peft
from peft import LoraConfig
from peft import TaskType
import random
import torch
from torch import nn
from torch.nn import functional
import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
from typing import Any, Dict, List, Literal, Tuple, Union

FLAGS = flags.FLAGS

class CharacterRepresentations(nn.Module):
    """
    PyTorch module that finds character representations from documents and
    calculates their similarity with attribute representations
    """

    def __init__(
            self,
            base_model: peft.PeftModel,
            alpha: float,
            trope_vecs: torch.tensor,
            trope_mask: torch.tensor,
            trait_vecs: torch.tensor):
        super().__init__()
        self.base_model = base_model
        self.alpha = alpha
        self.trope_vecs = trope_vecs
        self.trope_mask = trope_mask
        self.trait_vecs = trait_vecs
        self.use_tropes = True
        self.hidden_size = self.base_model.config.hidden_size
        self.trope_token_weight = nn.Linear(self.hidden_size, 1)
        self.character_token_weight = nn.Linear(self.hidden_size, 1)
        self.document_token_weight = nn.Linear(self.hidden_size, 1)
        self.character_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.document_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.chardoc_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.attribute_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def compute_attn_weighted_vecs(
            self,
            token_weight: nn.Linear,
            token_vecs: torch.tensor,
            mask: torch.tensor):
        """
        Compute attention-weighted representation
        """
        token_scores = token_weight(token_vecs).squeeze(dim=-1)
        token_attn = torch.softmax(token_scores + torch.log(mask), dim=-1)
        token_attn = token_attn.unsqueeze(dim=1)
        vecs = torch.matmul(token_attn, token_vecs).squeeze(dim=1)
        return vecs

    def forward(
            self,
            input_ids: torch.tensor,
            attention_mask: torch.tensor,
            character_mask: torch.tensor,
            attribute_ix: torch.tensor,
            labels: Union[torch.tensor, None] = None) -> torch.tensor:
        """
        Forward propagation of character representations model
        """

        # Encode document
        # batch x seqlen x hidden size
        token_vecs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask).last_hidden_state

        # Find character vectors
        # batch x hidden size
        character_vecs = self.compute_attn_weighted_vecs(
            self.character_token_weight, token_vecs, character_mask)

        # Find document vectors
        # batch x hidden size
        document_vecs = self.compute_attn_weighted_vecs(
            self.document_token_weight, token_vecs, attention_mask)

        # Project character and document vectors
        character_vecs = self.character_proj(character_vecs)
        document_vecs = self.document_proj(document_vecs)

        # Linear combination of character and document vecs
        # batch x hidden size
        chardoc_vecs = (
            self.alpha * character_vecs + (1 - self.alpha) * document_vecs)

        # Project chardoc vectors
        chardoc_vecs = self.character_proj(character_vecs)

        # normalize chardoc vectors
        chardoc_vecs = functional.normalize(chardoc_vecs, p=2, dim=-1)

        if self.use_tropes:

            # Index into trope vectors
            # batch x trope seqlen x hidden size
            trope_token_vecs = self.trope_vecs[attribute_ix]
            trope_mask = self.trope_mask[attribute_ix]

            # Find trope vectors
            # batch x hidden size
            trope_vecs = self.compute_attn_weighted_vecs(
                self.trope_token_weight, trope_token_vecs, trope_mask)
            
            # Project trope vecs
            trope_vecs = self.attribute_proj(trope_vecs)

            # Normalize trope vecs
            trope_vecs = functional.normalize(trope_vecs, p=2, dim=-1)

            # Find dot similarity
            sims = torch.sum(chardoc_vecs * trope_vecs, dim=-1)

            # Binary cross entropy if using tropes
            loss_fn = functional.binary_cross_entropy_with_logits

        else:

            # Index into trait vectors
            # batch x nclasses x hidden size
            trait_vecs = self.trait_vecs[attribute_ix]

            # Project trait vectors
            trait_vecs = self.attribute_proj(trait_vecs)

            # Normalize trait vecs
            trait_vecs = functional.normalize(trait_vecs, p=2, dim=-1)

            # Find dot similarity
            sims = torch.sum(chardoc_vecs.unsqueeze(dim=1) * trait_vecs, dim=-1)

            # Cross entropy for traits
            loss_fn = functional.cross_entropy

        # Return loss when training, and logits when inferring
        if labels is not None:
            loss = loss_fn(sims, labels)
            return loss
        else:
            return sims

def encode_tropes_and_traits(
        partial_state: PartialState,
        attributes_dir: str,
        trope_definitions: List[str],
        traits: List[str],
        tokenizer: AutoTokenizer,
        model: AutoModel) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode trope definitions and traits
    """

    # Define the logging function
    def log(message: str):
        if partial_state.is_local_main_process:
            logging.info(message)

    # Tokenize tropes
    log("tokenizing tropes")
    trope_encoding = tokenizer(
        trope_definitions,
        padding="max_length",
        max_length=data.TROPE_SEQLEN,
        return_tensors="pt")
    trope_input_ids = trope_encoding["input_ids"]
    trope_attention_mask = trope_encoding["attention_mask"]

    # Tokenize traits
    log("tokenizing traits")
    trait_encoding = tokenizer(
        traits,
        padding="longest",
        return_tensors="pt")
    trait_input_ids = trait_encoding["input_ids"]
    trait_attention_mask = trait_encoding["attention_mask"]

    # Encode tropes and traits
    with torch.no_grad():

        # Encode tropes in batches
        n_batches = math.ceil(
            len(trope_definitions)/data.TROPE_ENCODING_BATCH_SIZE)
        batch_vecs_list = []

        for i in tqdm.trange(
                n_batches,
                desc="encoding tropes",
                disable=not partial_state.is_local_main_process):
            batch_input_ids = trope_input_ids[
                i * data.TROPE_ENCODING_BATCH_SIZE:
                (i + 1) * data.TROPE_ENCODING_BATCH_SIZE]
            batch_attention_mask = trope_attention_mask[
                i * data.TROPE_ENCODING_BATCH_SIZE:
                (i + 1) * data.TROPE_ENCODING_BATCH_SIZE]
            batch_vecs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask).last_hidden_state
            batch_vecs_list.append(batch_vecs)
        trope_vecs = torch.cat(batch_vecs_list, dim=0).to(partial_state.device)

        # Encode traits
        log("encoding traits")
        trait_vecs = model(
            input_ids=trait_input_ids, attention_mask=trait_attention_mask)
        trait_vecs[trait_attention_mask == 0] = 0
        trait_vecs = (
            trait_vecs.sum(dim=1)
            /trait_attention_mask.sum(dim=-1, keepdim=True))

    # Save tensors
    if partial_state.is_local_main_process:
        log("saving trope and trait vectors")
        os.makedirs(attributes_dir, exist_ok=True)
        trope_vecs_file = os.path.join(attributes_dir, "trope-vecs.pt")
        trope_attention_mask_file = os.path.join(
            attributes_dir, "trope-mask.pt")
        trait_vecs_file = os.path.join(attributes_dir, "trait-vecs.pt")
        torch.save(trope_vecs, trope_vecs_file)
        torch.save(trope_attention_mask, trope_attention_mask_file)
        torch.save(trait_vecs, trait_vecs_file)

    return trope_vecs, trope_attention_mask, trait_vecs

class CustomDataCollatorWithPadding:
    """
    Custom data collator. Pad input ids and character mask to max length in
    batch and create attention mask
    """

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Retrieve the batch vectors
        batch_input_ids = []
        batch_character_mask = []
        batch_attribute_ix = []
        batch_labels = []

        for sample in batch:
            batch_input_ids.append(sample["input_ids"])
            batch_character_mask.append(sample["character_mask"])
            batch_attribute_ix.append(sample["attribute_ix"])
            batch_labels.append(sample["labels"])

        # Find the maximum sequence length in batch
        maxlen = max(map(len, batch_input_ids))

        # Pad sequences
        batch_attention_mask = []
        for i in range(len(batch_input_ids)):
            n = len(batch_input_ids[i])
            batch_input_ids[i] = (
                batch_input_ids[i]
                + (maxlen - n) * [self.tokenizer.pad_token_id])
            batch_character_mask[i] = (
                batch_character_mask[i] + (maxlen - n) * [0])
            batch_attention_mask.append([1] * n + [0] * (maxlen - n))

        # Tensorize
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_mask = torch.tensor(
            batch_attention_mask, dtype=torch.long)
        batch_character_mask = torch.tensor(batch_character_mask)
        batch_attribute_ix = torch.tensor(batch_attribute_ix)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "character_mask": batch_character_mask,
            "attribute_ix": batch_attribute_ix,
            "labels": batch_labels
        }

def train(partial_state: PartialState,
          experiments_dir: str,
          attributes_dir: str,
          train_data: List[Dict[str, Union[str, int]]],
          dev_data: List[Dict[str, Union[str, int]]],
          tropes: List[str],
          trope_definitions: List[str],
          traits: List[str],
          train_and_dev_dataset_name:
            Literal["chatter-contexts", "personet"] = "chatter-contexts"):
    """Character representation modeling training"""

    # Define the logging function
    def log(message: str):
        if partial_state.is_local_main_process:
            logging.info(message)

    # Instantiate quantization config
    compute_dtype = torch.bfloat16 if FLAGS.bf16 else torch.float16
    if FLAGS.load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_storage=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True)
    elif FLAGS.load_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    # Instantiate LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=FLAGS.lora_target_module,
        modules_to_save=["embed_tokens"],
        r=FLAGS.rank,
        lora_alpha=FLAGS.alpha,
        lora_dropout=FLAGS.dropout,
        use_rslora=True,
        bias="none")

    # Instantiate base model
    log("instantiating base model")
    base_model = AutoModel.from_pretrained(
        FLAGS.modelname,
        dtype=compute_dtype,
        quantization_config=quantization_config,
        device_map={"": partial_state.process_index},
        attn_implementation=FLAGS.attn)

    # Wrapping the base model into LoRA
    log("wrapping base model with LoRA")
    base_model = peft.prepare_model_for_kbit_training(base_model)
    base_model = peft.get_peft_model(base_model, lora_config)
    n_trainable, n_all = base_model.get_nb_trainable_parameters()
    log(f"{n_trainable} trainable params, {n_all} all params, "
        f"{100*n_trainable/n_all:.1f}%trainable")

    # Instantiating tokenizer
    log("instantiating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)

    # Setting pad token id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        base_model.config.pad_token_id = tokenizer.eos_token_id
        base_model.config.eos_token_id = tokenizer.eos_token_id
        base_model.generation_config.pad_token_id = tokenizer.eos_token_id
        base_model.generation_config.eos_token_id = tokenizer.eos_token_id

    # Read tropes and traits vectors if they have already been encoded
    # or encode them
    if os.path.exists(attributes_dir):

        # Read the tropes and traits vectors
        trope_vecs_file = os.path.join(attributes_dir, "trope-vecs.pt")
        trope_attention_mask_file = os.path.join(
            attributes_dir, "trope-mask.pt")
        trait_vecs_file = os.path.join(attributes_dir, "trait-vecs.pt")
        trope_vecs = torch.load(
            trope_vecs_file,
            weights_only=True,
            map_location=partial_state.device)
        trope_mask = torch.load(
            trope_attention_mask_file,
            weights_only=True,
            map_location=partial_state.device)
        trait_vecs = torch.load(
            trait_vecs_file,
            weights_only=True,
            map_location=partial_state.device)
    else:

        # Encode tropes and traits
        trope_vecs, trope_mask, trait_vecs = encode_tropes_and_traits(
            partial_state=partial_state,
            attributes_dir=attributes_dir,
            trope_definitions=trope_definitions,
            traits=traits,
            tokenizer=tokenizer,
            model=base_model)

        # Move to cuda device
        trope_vecs = trope_vecs.to(partial_state.device)
        trope_mask = trope_mask.to(partial_state.device)
        trait_vecs = trait_vecs.to(partial_state.device)

    # Instantiate character representations model
    log("instantiating character representations")
    model = CharacterRepresentations(
        base_model=base_model,
        alpha=FLAGS.alpha,
        trope_vecs=trope_vecs,
        trope_mask=trope_mask,
        trait_vecs=trait_vecs)

    # Create tropes and traits to index mapping
    tropes_ix = {trope: i for i, trope in enumerate(tropes)}
    traits_ix = {trait: i for i, trait in enumerate(traits)}

    # create datasets
    log("creating datasets")
    random.shuffle(train_data)
    train_dataset, train_df = data.create_crm_dataset(
        data=train_data,
        tropes_ix=tropes_ix,
        traits_ix=traits_ix,
        tokenizer=tokenizer,
        dataset_name=train_and_dev_dataset_name,
        disable_progress_bar=not partial_state.is_local_main_process)
    dev_dataset, dev_df = data.create_crm_dataset(
        data=dev_data,
        tropes_ix=tropes_ix,
        traits_ix=traits_ix,
        tokenizer=tokenizer,
        dataset_name=train_and_dev_dataset_name,
        disable_progress_bar=not partial_state.is_local_main_process)

    # log size of sequences
    log("counting tokens")
    train_ntokens = list(map(len, train_dataset["input_ids"]))
    dev_ntokens = list(map(len, dev_dataset["input_ids"]))
    if partial_state.is_local_main_process:
        logging.info("\ntokens/sample:")
        logging.info(
            f"{train_and_dev_dataset_name} train: "
            f"max = {max(train_ntokens)}, "
            f"min = {min(train_ntokens)}, "
            f"95%tile = {np.quantile(train_ntokens, 0.95):.1f}")
        logging.info(
            f"{train_and_dev_dataset_name} dev: "
            f"max = {max(dev_ntokens)}, "
            f"min = {min(dev_ntokens)}, "
            f"95%tile = {np.quantile(dev_ntokens, 0.95):.1f}")

    # create Training Args
    training_args = TrainingArguments(
        output_dir=experiments_dir,
        eval_on_start=FLAGS.eval_on_start,
        eval_strategy="steps" if FLAGS.eval else "no",
        eval_steps=FLAGS.eval_steps,
        eval_delay=FLAGS.eval_delay,
        eval_accumulation_steps=FLAGS.eval_accumulation_steps,
        per_device_train_batch_size=FLAGS.train_batch_size,
        per_device_eval_batch_size=FLAGS.eval_batch_size,
        learning_rate=FLAGS.lr,
        gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
        warmup_steps=FLAGS.warmup_steps,
        weight_decay=FLAGS.weight_decay,
        max_grad_norm=FLAGS.max_grad_norm,
        max_steps=FLAGS.train_steps,
        seed=2025,
        data_seed=2025,
        logging_strategy="steps",
        logging_steps=FLAGS.logging_steps,
        bf16=FLAGS.bf16,
        fp16=not FLAGS.bf16,
        optim=FLAGS.optim,
        gradient_checkpointing=FLAGS.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        save_strategy="no")

    # set up the logger callback
    if partial_state.is_local_main_process:
        callbacks = [utils.LoggerCallback(experiments_dir)]
    else:
        callbacks = None

    # create trainer
    log("instantiating trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=CustomDataCollatorWithPadding(tokenizer=tokenizer),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=callbacks)

    # train
    log("training started")
    log("=" * 80)
    trainer.train()
    log("=" * 80)
    log("training done")

    # save model
    if FLAGS.save_model:
        partial_state.wait_for_everyone()
        log("saving model")
        trainer.save_model(experiments_dir)

def predict(modelpath: str, datasetname_to_dataset: Dict[str, List]):
    """Run inference on datasets using character representations models"""