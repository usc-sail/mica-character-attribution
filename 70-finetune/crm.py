"""Training and prediction functions for character representations modeling"""

from absl import flags
from absl import logging
from accelerate import PartialState
from peft import LoraConfig
from peft import TaskType
import torch
from transformers import AutoModel
from transformers import BitsAndBytesConfig
from typing import Dict, List, Literal, Union

FLAGS = flags.FLAGS

def train(partial_state: PartialState,
          experiments_dir: str,
          train_data: List[Dict[str, Union[str, int]]],
          dev_data: List[Dict[str, Union[str, int]]],
          chatter_test_data: List[Dict[str, Union[str, int]]],
          personet_test_data: List[Dict[str, Union[str, int]]],
          train_and_dev_dataset_name: Literal["chatter-contexts", "personet"] = "chatter-contexts"):
    """Character representation modeling training"""

    # define the logging function
    def log(message: str):
        if partial_state.is_local_main_process:
            logging.info(message)

    # instantiate quantization config
    compute_dtype = torch.bfloat16 if FLAGS.bf16 else torch.float16
    if FLAGS.load_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=compute_dtype,
                                                 bnb_4bit_quant_storage=compute_dtype,
                                                 bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_use_double_quant=True)
    elif FLAGS.load_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    # instantiate LoRA config
    lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                             target_modules=FLAGS.lora_target_module,
                             modules_to_save=["embed_tokens"],
                             r=FLAGS.rank,
                             lora_alpha=FLAGS.alpha,
                             lora_dropout=FLAGS.dropout,
                             use_rslora=True,
                             bias="none")

    # instantiate model
    log("instantiating model")
    if FLAGS.model == "sft":
        model = AutoModelForCausalLM.from_pretrained(FLAGS.modelname,
                                                     torch_dtype=compute_dtype,
                                                     quantization_config=quantization_config,
                                                     device_map={"": PARTIALSTATE.process_index},
                                                     attn_implementation=FLAGS.attn)
    else:
        model = models.BinaryClassifier.from_pretrained(FLAGS.model,
                                                            num_labels=2,
                                                            torch_dtype=compute_dtype,
                                                            quantization_config=quantization_config,
                                                            device_map={"": PARTIALSTATE.process_index},
                                                            attn_implementation=FLAGS.attn)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id

    # instantiating tokenizer
    log("\n\ninstantiating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # add special tokens (only done for classification method)
    if not FLAGS.instrtune:
        log("adding special tokens")
        tokenizer.add_special_tokens({"additional_special_tokens": [data.CHARACTER_TOKEN,
                                                                    data.ATTRIBUTE_TOKEN,
                                                                    data.CONTEXT_TOKEN]},
                                     replace_additional_special_tokens=False)
        model.resize_token_embeddings(len(tokenizer.vocab))
        if FLAGS.multiclass:
            model.attribute_token_id = tokenizer.vocab[data.ATTRIBUTE_TOKEN]

        # create LoRA model (only done for classification method)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        n_trainable, n_all = model.get_nb_trainable_parameters()
        log(f"{n_trainable} trainable params, {n_all} all params, {100*n_trainable/n_all:.1f}%trainable")
    log("\n\n")

    # create datasets
    log("creating datasets")
    random.shuffle(train_data)
    kwargs = dict(tokenizer=tokenizer,
                  instrtune=FLAGS.instrtune,
                  multiclass=FLAGS.multiclass,
                  batch_size=FLAGS.dataset_batch_size,
                  disable_progress_bar=not PARTIALSTATE.is_local_main_process)
    train_dataset, _ = data.create_dataset(data=train_data,
                                                 name="train",
                                                 instr_seqlen=train_instr_seqlen,
                                                 **kwargs)
    chatter_dev_dataset, chatter_dev_df = data.create_dataset(data=chatter_dev_data,
                                                                    name="chatter dev",
                                                                    instr_seqlen=FLAGS.chatter_instr_seqlen,
                                                                    **kwargs)
    chatter_test_dataset, chatter_test_df = data.create_dataset(data=chatter_test_data,
                                                                      name="chatter test",
                                                                      instr_seqlen=FLAGS.chatter_instr_seqlen,
                                                                      **kwargs)
    anonymized_chatter_dev_dataset, anonymized_chatter_dev_df = data.create_dataset(
        data=anonymized_chatter_dev_data,
        name="anonymized chatter dev",
        instr_seqlen=FLAGS.chatter_instr_seqlen,
        **kwargs)
    anonymized_chatter_test_dataset, anonymized_chatter_test_df = data.create_dataset(
        data=anonymized_chatter_test_data,
        name="anonymized chatter test",
        instr_seqlen=FLAGS.chatter_instr_seqlen,
        **kwargs)
    personet_dev_dataset, personet_dev_df = data.create_dataset(data=personet_dev_data,
                                                                      name="personet dev",
                                                                      instr_seqlen=FLAGS.personet_instr_seqlen,
                                                                      **kwargs)
    personet_test_dataset, personet_test_df = data.create_dataset(data=personet_test_data,
                                                                        name="personet test",
                                                                        instr_seqlen=FLAGS.personet_instr_seqlen,
                                                                        **kwargs)
    train_ntokens = list(map(len, train_dataset["input_ids"]))
    chatter_dev_ntokens = list(map(len, chatter_dev_dataset["input_ids"]))
    chatter_test_ntokens = list(map(len, chatter_test_dataset["input_ids"]))
    anonymized_chatter_dev_ntokens = list(map(len, anonymized_chatter_dev_dataset["input_ids"]))
    anonymized_chatter_test_ntokens = list(map(len, anonymized_chatter_test_dataset["input_ids"]))
    personet_dev_ntokens = list(map(len, personet_dev_dataset["input_ids"]))
    personet_test_ntokens = list(map(len, personet_test_dataset["input_ids"]))
    if PARTIALSTATE.is_local_main_process:
        logging.info("\ntokens/sample:")
        logging.info(f"{FLAGS.train_dataset.upper()} train: max = {max(train_ntokens)}, min = {min(train_ntokens)}, "
                     f"95%tile = {np.quantile(train_ntokens, 0.95):.1f}")
        logging.info(f"CHATTER dev: max = {max(chatter_dev_ntokens)}, min = {min(chatter_dev_ntokens)}, "
                     f"95%tile = {np.quantile(chatter_dev_ntokens, 0.95):.1f}")
        logging.info(f"CHATTER test: max = {max(chatter_test_ntokens)}, min = {min(chatter_test_ntokens)}, "
                     f"95%tile = {np.quantile(chatter_test_ntokens, 0.95):.1f}")
        logging.info(f"ANONYMIZED CHATTER dev: max = {max(anonymized_chatter_dev_ntokens)}, "
                     f"min = {min(anonymized_chatter_dev_ntokens)}, "
                     f"95%tile = {np.quantile(anonymized_chatter_dev_ntokens, 0.95):.1f}")
        logging.info(f"ANONYMIZED CHATTER test: max = {max(anonymized_chatter_test_ntokens)}, "
                     f"min = {min(anonymized_chatter_test_ntokens)}, "
                     f"95%tile = {np.quantile(anonymized_chatter_test_ntokens, 0.95):.1f}")
        logging.info(f"PERSONET dev: max = {max(personet_dev_ntokens)}, min = {min(personet_dev_ntokens)}, "
                     f"95%tile = {np.quantile(personet_dev_ntokens, 0.95):.1f}")
        logging.info(f"PERSONET test: max = {max(personet_test_ntokens)}, min = {min(personet_test_ntokens)}, "
                     f"95%tile = {np.quantile(personet_test_ntokens, 0.95):.1f}")
        logging.info("\n\n")

    # create SFT config or Training Args
    kwargs = dict(output_dir=experiments_dir,
                  eval_strategy="steps" if FLAGS.eval else "no",
                  eval_steps=FLAGS.eval_steps,
                  eval_delay=FLAGS.eval_delay,
                  eval_accumulation_steps=FLAGS.eval_accumulation_steps,
                  per_device_train_batch_size=FLAGS.train_batch_size,
                  per_device_eval_batch_size=FLAGS.eval_batch_size,
                  learning_rate=FLAGS.lr,
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
                  gradient_checkpointing_kwargs={"use_reentrant": False},
                  save_strategy="no")
    if FLAGS.instrtune:
        config = SFTConfig(max_seq_length=train_instr_seqlen,
                           packing=False,
                           **kwargs)
    else:
        config = TrainingArguments(**kwargs)

    # create compute metrics instance
    compute_metrics = evaluate.ComputeMetrics(tokenizer)
    compute_metrics.multiclass = FLAGS.multiclass
    if FLAGS.train_dataset == "chatter":
        compute_metrics.eval_df = chatter_dev_df
        compute_metrics.set_dataset("chatter")
    else:
        compute_metrics.eval_df = personet_dev_df
        compute_metrics.set_dataset("personet")

    # create trainer
    log("instantiating trainer")
    if FLAGS.instrtune:
        trainer = SFTTrainer(model=model,
                             args=config,
                             data_collator=DataCollatorForCompletionOnlyLM(response_template=data.ANSWER_TEMPLATE,
                                                                           tokenizer=tokenizer),
                             train_dataset=train_dataset,
                             eval_dataset=(chatter_dev_dataset
                                           if FLAGS.train_dataset == "chatter" else personet_dev_dataset),
                             processing_class=tokenizer,
                             peft_config=lora_config,
                             preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                             compute_metrics=compute_metrics.compute_instruction_metrics,
                             callbacks=[callback])
    else:
        trainer = Trainer(model=model,
                          args=config,
                          data_collator=DataCollatorWithPadding(tokenizer=tokenizer,
                                                                padding="longest",
                                                                return_tensors="pt"),
                          train_dataset=train_dataset,
                          eval_dataset=(chatter_dev_dataset
                                        if FLAGS.train_dataset == "chatter" else personet_dev_dataset),
                          processing_class=tokenizer,
                          compute_metrics=compute_metrics.compute_classification_metrics,
                          callbacks=[callback])

    # train
    log("\n\n\ntraining started")
    log("================================================================================================")
    trainer.train()
    log("\n\n\n==========================================================================================")
    log("training done\n\n\n")

    # plot train loss, dev loss, and dev metric
    log("plotting\n\n")
    if PARTIALSTATE.is_local_main_process:
        with jsonlines.open(logs_file) as reader:
            logs = list(reader)
        metric = "eval_accuracy" if FLAGS.train_dataset == "chatter" else "eval_accuracy@1"
        plot_logs(logs, metric, experiments_dir)

    # predict and evaluate
    test_datasets = [chatter_dev_dataset, chatter_test_dataset, anonymized_chatter_dev_dataset,
                     anonymized_chatter_test_dataset, personet_dev_dataset, personet_test_dataset]
    test_dfs = [chatter_dev_df, chatter_test_df, anonymized_chatter_dev_df, anonymized_chatter_test_df,
                personet_dev_df, personet_test_df]
    test_names = ["CHATTER dev", "CHATTER test", "ANONYMIZED CHATTER dev", "ANONYMIZED CHATTER test",
                  "PERSONET dev", "PERSONET test"]
    for test_dataset, test_df, test_name in zip(test_datasets, test_dfs, test_names):
        log(f"evaluating {test_name}")
        if "CHATTER" in test_name:
            compute_metrics.set_dataset("chatter")
        else:
            compute_metrics.set_dataset("personet")
        compute_metrics.eval_df = test_df
        test_output = trainer.predict(test_dataset)
        log(f"{test_output.metrics}\n\n")
        test_filename = test_name.replace(" ", "_")
        test_predictions_file = os.path.join(experiments_dir, test_filename + ".csv")
        test_metrics_file = os.path.join(experiments_dir, test_filename + ".json")
        if PARTIALSTATE.is_local_main_process:
            test_df.to_csv(test_predictions_file, index=False)
            json.dump(test_output.metrics, open(test_metrics_file, "w"))

    PARTIALSTATE.wait_for_everyone()
    # save model
    if FLAGS.save_model:
        log("saving model")
        trainer.save_model(experiments_dir)

def predict(modelpath: str, datasetname_to_dataset: Dict[str, List]):
    """Run inference on datasets using character representations models"""