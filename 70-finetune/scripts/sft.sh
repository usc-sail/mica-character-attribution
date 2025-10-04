#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1

# main training command
CMD="accelerate launch 
--config_file configs/deepspeed.yaml \
train.py \
--model sft \
--modelname meta-llama/Llama-3.1-8B-Instruct \
--load_4bit \
--attn flash_attention_2 \
--train_batch_size 1 \
--lora_target_module q_proj \
--lora_target_module k_proj \
--lora_target_module v_proj \
--rank 4 \
--alpha 8 \
--lr 2e-5 \
--eval_batch_size 4 \
--eval_delay 128 \
--logging_steps 32 \ 
--save_model \
--alsologtostderr \
--noshowprefixforinfo"

# train on chatter-contexts
$CMD \
--train_dataset chatter-contexts \
--chatter_train_and_dev_truncation_strategy semantic \
--chatter_train_and_dev_size 2000 \
--train_steps 1024