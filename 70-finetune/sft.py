"""Training and prediction functions for supervized fine-tuning"""
import data
import evaluate
import utils

from absl import flags
from absl import logging
from accelerate import PartialState
from accelerate.utils import gather_object
import json
import jsonlines
import math
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
        train_and_dev_dataset_name: 
            Literal["chatter-contexts", "personet"] = "chatter-contexts"):
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
    dev_dataset, _ = data.create_sft_dataset(
        data=dev_data,
        tokenizer=tokenizer,
        dataset_name=train_and_dev_dataset_name,
        chatter_contexts_size_in_words=int(FLAGS.chatter_size),
        tokenization_batch_size=FLAGS.tokenization_batch_size,
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

    sft_config = SFTConfig(
        output_dir=experiments_dir,
        max_length=None,
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
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        data_collator=DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
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

    # plot train loss and dev loss, and dev metric
    log("plotting")
    if partial_state.is_local_main_process:
        with jsonlines.open(callbacks[0].logs_file) as reader:
            logs = list(reader)
        utils.plot_logs(logs, "eval_mean_token_accuracy", experiments_dir)

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
        size = (int(dataset_name.split("-")[4])
                if matched_dataset_name == "chatter-contexts" else 2000)
        dataset, df = data.create_sft_dataset(
            data=datalist,
            tokenizer=tokenizer,
            only_prompt=True,
            dataset_name=matched_dataset_name,
            chatter_contexts_size_in_words=size,
            tokenization_batch_size=FLAGS.tokenization_batch_size,
            disable_progress_bar=not partial_state.is_local_main_process)
        dataset_name_to_dataset_and_df[dataset_name] = (dataset, df)

    # Create Compute Metrics instance
    compute_metrics = evaluate.ComputeMetrics()

    # Create predictions directory
    predictions_dir = os.path.join(FLAGS.modelpath, "predictions")
    if partial_state.is_local_main_process:
        os.makedirs(predictions_dir, exist_ok=True)

    # Sort dataset names so all processes see the same order
    dataset_names = sorted(list(dataset_name_to_dataset_and_df.keys()))

    # Loop over each dataset and generate
    for dataset_name in dataset_names:
        dataset, df = dataset_name_to_dataset_and_df[dataset_name]
        log(f"evaluating {dataset_name}")
        matched_dataset_name = re.match(
            "(chatter-contexts)|(chatter-segments)|(personet)",
            dataset_name).group(0)
        log(f"{len(dataset)} samples")

        partial_state.wait_for_everyone()

        # Split the dataset between the GPUs
        input_ids = dataset["input_ids"]
        with partial_state.split_between_processes(
                input_ids) as process_input_ids:
            logging.info(
                f"PROCESS {partial_state.process_index}: "
                f"{len(process_input_ids)} samples")

            # Split the per-GPU dataset into batches
            n_batches = math.ceil(
                len(process_input_ids)/FLAGS.prediction_batch_size)

            # Collect the per-GPU responses
            process_responses = []

            # Loop over each batch
            for i in range(n_batches):

                # Pad the batch to the longest sequence in the batch
                batch_input_ids = process_input_ids[
                    i * FLAGS.prediction_batch_size:
                    (i + 1) * FLAGS.prediction_batch_size]
                maxlen = max(map(len, batch_input_ids))
                padded_batch_input_ids = []
                batch_attention_mask = []
                for input_ids in batch_input_ids:
                    padded_input_ids = (
                        [tokenizer.pad_token_id] * (maxlen - len(input_ids))
                        + input_ids)
                    attention_mask = (
                        [0] * (maxlen - len(input_ids)) + [1] * len(input_ids))
                    padded_batch_input_ids.append(padded_input_ids)
                    batch_attention_mask.append(attention_mask)

                # Convert to torch tensor and move to GPU
                padded_batch_input_ids = (
                    torch.tensor(padded_batch_input_ids)
                    .to(partial_state.device))
                batch_attention_mask = (
                    torch.tensor(batch_attention_mask).to(partial_state.device))

                # Generate response
                batch_output_ids = model.generate(
                    input_ids=padded_batch_input_ids,
                    attention_mask=batch_attention_mask,
                    do_sample=FLAGS.do_sample,
                    top_k=FLAGS.top_k,
                    top_p=FLAGS.top_p,
                    temperature=FLAGS.temperature,
                    max_new_tokens=FLAGS.max_output_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    return_legacy_cache=True)

                # Keep the generated output ids
                batch_output_ids = batch_output_ids[
                    :, padded_batch_input_ids.shape[1]:]

                # Decode the generated output ids
                batch_responses = tokenizer.batch_decode(
                    batch_output_ids, skip_special_tokens=True)
                batch_responses = [response.strip().lower()
                                    for response in batch_responses]

                # Collect the per-GPU responses
                process_responses += batch_responses
                logging.info(
                    f"PROCESS {partial_state.process_index}: "
                    f"Batch {i + 1} / {n_batches} done")

            # Contain in list for gather
            responses = [process_responses]

        # Gather per-GPU responses
        responses_arr = gather_object(responses)
        responses = [
            response for responses in responses_arr for response in responses]
        df["pred-text"] = responses[:len(dataset)]

        # Convert label text to label
        preds = []
        for _, row in df.iterrows():
            pred = np.nan
            if (matched_dataset_name == "chatter-contexts"
                    or matched_dataset_name == "chatter-segments"):
                mobj = re.search(r"\w+", row["pred-text"])
                if mobj is not None:
                    if mobj.group(0) == "yes":
                        pred = 1
                    elif mobj.group(0) == "no":
                        pred = 0
            else:
                traits = [row[f"attribute-{i + 1}"]
                          for i in range(data.NCLASSES)]
                for trait in traits:
                    if trait in row["pred-text"]:
                        pred = trait
            preds.append(pred)
        df["pred"] = preds

        # Compute metrics
        metrics = compute_metrics(matched_dataset_name, df)

        # Print and save metrics
        log(f"{dataset_name}\n{metrics}\n")
        if partial_state.is_local_main_process:
            predictions_file = os.path.join(
                predictions_dir,
                f"{dataset_name}.csv")
            metrics_file = os.path.join(
                predictions_dir,
                f"{dataset_name}.json")
            df.to_csv(predictions_file, index=False)
            json.dump(metrics, open(metrics_file, "w"))

        # Wait for each process to complete dataset operations
        partial_state.wait_for_everyone()