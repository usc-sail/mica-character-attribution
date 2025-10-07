"""Training and prediction functions for supervized fine-tuning"""
import data
import evaluate
import utils

from absl import flags
from absl import logging
from accelerate import PartialState
import json
import jsonlines
import numpy as np
import os
from peft import LoraConfig
from peft import PeftModel
from peft import TaskType
import random
import re
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import Trainer
from trl import SFTConfig
from trl import SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from typing import Dict, List, Literal, Union

FLAGS = flags.FLAGS

def preprocess_logits_for_metrics(logits: torch.Tensor, _) -> torch.Tensor:
    """
    Preprocess logits to prepare for metrics computation for the supervized 
    fine-tuning method. We use this functions to save space primarily, 
    converting batch-size x seqlen x vocab-size tensors to 
    batch-size x seqlen x 2 tensors
    """
    return logits.argmax(dim=-1).int() # batch-size x seqlen

def train(
        partial_state: PartialState,
        experiments_dir: str,
        train_data: List[Dict[str, Union[str, int]]],
        dev_data: List[Dict[str, Union[str, int]]],
        chatter_test_data: List[Dict[str, Union[str, int]]],
        personet_test_data: List[Dict[str, Union[str, int]]],
        train_and_dev_dataset_name: Literal["chatter-contexts", "personet"] = "chatter-contexts"):
    """
    Supervized fine-tuning training
    """

    # define the logging function
    def log(message: str):
        if partial_state.is_local_main_process:
            logging.info(message)

    # instantiate quantization config
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

    # instantiate LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=FLAGS.lora_target_module,
        modules_to_save=["embed_tokens", "lm_head"],
        r=FLAGS.rank,
        lora_alpha=FLAGS.alpha,
        lora_dropout=FLAGS.dropout,
        use_rslora=True,
        bias="none")

    # instantiate model
    log("instantiating model")
    model = AutoModelForCausalLM.from_pretrained(
        FLAGS.modelname,
        dtype=compute_dtype,
        quantization_config=quantization_config,
        device_map={"": partial_state.process_index},
        attn_implementation=FLAGS.attn)

    # instantiating tokenizer
    log("instantiating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.modelname)

    # setting pad token id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    # create datasets
    log("creating datasets")
    random.shuffle(train_data)
    train_dataset, _ = data.create_sft_dataset(
        data=train_data,
        tokenizer=tokenizer,
        dataset_name=train_and_dev_dataset_name,
        chatter_contexts_size_in_words=int(FLAGS.chatter_size),
        tokenization_batch_size=FLAGS.tokenization_batch_size,
        disable_progress_bar=not partial_state.is_local_main_process)
    dev_dataset, dev_df = data.create_sft_dataset(
        data=dev_data,
        tokenizer=tokenizer,
        dataset_name=train_and_dev_dataset_name,
        chatter_contexts_size_in_words=int(FLAGS.chatter_size),
        tokenization_batch_size=FLAGS.tokenization_batch_size,
        disable_progress_bar=not partial_state.is_local_main_process)
    chatter_test_dataset, chatter_test_df = data.create_sft_dataset(
        data=chatter_test_data,
        tokenizer=tokenizer,
        dataset_name="chatter-contexts",
        chatter_contexts_size_in_words=int(FLAGS.chatter_size),
        tokenization_batch_size=FLAGS.tokenization_batch_size,
        disable_progress_bar=not partial_state.is_local_main_process)
    personet_test_dataset, personet_test_df = data.create_sft_dataset(
        data=personet_test_data,
        tokenizer=tokenizer,
        dataset_name="personet",
        tokenization_batch_size=FLAGS.tokenization_batch_size,
        disable_progress_bar=not partial_state.is_local_main_process)
    
    # log size of sequences
    train_ntokens = list(map(len, train_dataset["input_ids"]))
    dev_ntokens = list(map(len, dev_dataset["input_ids"]))
    chatter_test_ntokens = list(map(len, chatter_test_dataset["input_ids"]))
    personet_test_ntokens = list(map(len, personet_test_dataset["input_ids"]))
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
        logging.info(
            "chatter test: "
            f"max = {max(chatter_test_ntokens)}, "
            f"min = {min(chatter_test_ntokens)}, "
            f"95%tile = {np.quantile(chatter_test_ntokens, 0.95):.1f}")
        logging.info(
            "personet test: "
            f"max = {max(personet_test_ntokens)}, "
            f"min = {min(personet_test_ntokens)}, "
            f"95%tile = {np.quantile(personet_test_ntokens, 0.95):.1f}\n")

    sft_config = SFTConfig(
        output_dir=experiments_dir,
        max_length=None,
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

    # create compute metrics instance
    compute_metrics = evaluate.ComputeMetrics(tokenizer)
    compute_metrics.eval_df = dev_df
    compute_metrics.dataset = FLAGS.train_dataset

    # set up the logger callback
    if partial_state.is_local_main_process:
        callbacks = [utils.LoggerCallback(experiments_dir)]
    else:
        callbacks = None

    # create trainer
    log("instantiating trainer")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        data_collator=DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics.compute_sft_metrics,
        callbacks=callbacks)
    

    # train
    log("training started")
    log("=" * 80)
    trainer.train()
    log("=" * 80)
    log("training done")

    # release training memory
    utils.release_training_memory(trainer)

    # plot train loss, dev loss, and dev metric
    log("plotting")
    if partial_state.is_local_main_process:
        with jsonlines.open(callbacks[0].logs_file) as reader:
            logs = list(reader)
        utils.plot_logs(logs, "eval_accuracy", experiments_dir)

    # predict and evaluate
    datasets = [dev_dataset, chatter_test_dataset, personet_test_dataset]
    dfs = [dev_df, chatter_test_df, personet_test_df]
    dataset_names = [train_and_dev_dataset_name, "chatter-contexts", "personet"]
    partitions = ["dev", "test", "test"]
    for dataset, df, name, partition in zip(
            datasets, dfs, dataset_names, partitions):
        log(f"evaluating {name} {partition}")
        compute_metrics.eval_df = df
        compute_metrics.dataset = name
        metrics = trainer.predict(dataset).metrics
        log(f"{metrics}\n\n")
        if partial_state.is_local_main_process:
            predictions_file = os.path.join(
                experiments_dir,
                f"{name}-{partition}.csv")
            metrics_file = os.path.join(
                experiments_dir,
                f"{name}-{partition}.json")
            df.to_csv(predictions_file, index=False)
            json.dump(metrics, open(metrics_file, "w"))

    # save model
    if FLAGS.save_model:
        partial_state.wait_for_everyone()
        log("saving model")
        trainer.save_model(experiments_dir)

def predict(
        partial_state: PartialState,
        datasetname_to_data: Dict[str, List]):
    """
    Run inference on datasets using SFT-trained LLMs
    """

    # define the logging function
    def log(message: str):
        if partial_state.is_local_main_process:
            logging.info(message)

    # instantiate quantization config
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

    # load model
    log("instantiating model")
    base_model = AutoModelForCausalLM.from_pretrained(
        FLAGS.modelname,
        dtype=compute_dtype,
        quantization_config=quantization_config,
        device_map={"": partial_state.process_index},
        attn_implementation=FLAGS.attn)
    model = PeftModel.from_pretrained(base_model, FLAGS.modelpath)

    # instantiating tokenizer
    log("instantiating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.modelname)

    # setting pad token id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    # create datasets
    log("creating datasets")
    dataset_name_to_dataset_and_df = {}
    for dataset_name, datalist in datasetname_to_data.items():
        matched_dataset_name = re.match(
            "(chatter-contexts)|(chatter-segments)|(personet)",
            dataset_name).group(0)
        size = int(dataset_name.split("-")[4]) if matched_dataset_name == "chatter-contexts" else 2000
        dataset, df = data.create_sft_dataset(
            data=datalist,
            tokenizer=tokenizer,
            dataset_name=matched_dataset_name,
            chatter_contexts_size_in_words=size,
            tokenization_batch_size=FLAGS.tokenization_batch_size,
            disable_progress_bar=not partial_state.is_local_main_process)
        dataset_name_to_dataset_and_df[dataset_name] = (dataset, df)

    # instantiate trainer; only used for prediction here
    log("instantiating trainer")
    predictions_dir = os.path.join(FLAGS.modelpath, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=predictions_dir,
        per_device_eval_batch_size=FLAGS.prediction_batch_size,
        bf16=FLAGS.bf16,
        fp16=not FLAGS.bf16)
    compute_metrics = evaluate.ComputeMetrics(tokenizer)
    dummy_train_dataset = list(dataset_name_to_dataset_and_df.values())[0][0]
    trainer = Trainer(
        model=model,
        args=sft_config,
        data_collator=DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id),
        train_dataset=dummy_train_dataset,
        processing_class=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics.compute_sft_metrics)

    # predict over datasets
    for dataset_name, (dataset, df) in dataset_name_to_dataset_and_df.items():
        log(f"evaluating {dataset_name}")
        matched_dataset_name = re.match(
            "(chatter-contexts)|(chatter-segments)|(personet)", dataset_name).group(0)
        compute_metrics.eval_df = df
        compute_metrics.dataset = matched_dataset_name
        metrics = trainer.predict(dataset).metrics
        log(f"{metrics}\n\n")
        if partial_state.is_local_main_process:
            predictions_file = os.path.join(
                FLAGS.modelpath,
                f"predictions/{dataset_name}.csv")
            metrics_file = os.path.join(
                FLAGS.modelpath,
                f"predictions/{dataset_name}.json")
            df.to_csv(predictions_file, index=False)
            json.dump(metrics, open(metrics_file, "w"))