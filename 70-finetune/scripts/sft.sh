#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1

# main training command
TRAIN="accelerate launch 
--config_file configs/deepspeed.yaml \
train.py \
--model sft \
--modelname meta-llama/Llama-3.1-8B-Instruct \
--load_4bit \
--attn flash_attention_2 \
--train_batch_size 2 \
--lora_target_module q_proj \
--lora_target_module k_proj \
--lora_target_module v_proj \
--rank 16 \
--alpha 32 \
--lr 2e-5 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing \
--eval_batch_size 4 \
--eval \
--eval_delay 0 \
--eval_on_start \
--eval_steps 32 \
--logging_steps 16 \ 
--save_model \
--alsologtostderr \
--noshowprefixforinfo"

# main prediction command
PREDICT="accelerate launch 
--config_file configs/deepspeed.yaml \
predict.py \
--model sft \
--modelname meta-llama/Llama-3.1-8B-Instruct \
--load_4bit \
--attn flash_attention_2
"

if [[ "$1" == "train" ]]; then

    # # train on chatter-contexts-semantic
    # $TRAIN \
    # --train_dataset chatter-contexts \
    # --chatter_truncation_strategy semantic \
    # --chatter_size 2000 \
    # --train_steps 1024

    # # train on chatter-contexts-first
    # $TRAIN \
    # --train_dataset chatter-contexts \
    # --chatter_truncation_strategy first \
    # --chatter_size 2000 \
    # --train_steps 1024

    # train on personet
    $TRAIN \
    --train_dataset personet \
    --train_steps 1024

else

    # predict on chatter-segments
    $PREDICT \
    --dataset chatter-segments-anonymized-test \
    --dataset chatter-segments-original-test \
    --modelpath $1 \
    --prediction_batch_size 1

    # predict on chatter-contexts & personet
    $PREDICT \
    --dataset chatter-contexts-original-semantic-2000-dev \
    --dataset chatter-contexts-original-semantic-2000-test \
    --dataset chatter-contexts-original-first-2000-dev \
    --dataset chatter-contexts-original-first-2000-test \
    --dataset personet-dev \
    --dataset personet-test \
    --modelpath $1 \
    --prediction_batch_size 4

fi