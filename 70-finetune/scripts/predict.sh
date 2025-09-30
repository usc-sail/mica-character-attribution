#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1

# main prediction command
CMD="accelerate launch --config_file configs/deepspeed.yaml predict.py \
--modelname meta-llama/Llama-3.1-8B-Instruct --load_4bit --attn flash_attention_2"

# # run inference on chatter-segments using SFT-trained on chatter-contexts-semantic-2000
# $CMD --modelpath /home/ubuntu/data/mica-character-attribution/70-finetune/sft-results/2000-trope-2025May15-174725 \
# --model sft \
# --dataset chatter-segments-original-test \
# --dataset chatter-segments-anonymized-test \
# --prediction_batch_size 1

# run inference on chatter-contexts-semantic-2000 & personet using SFT-trained on chatter-contexts-semantic-2000
$CMD --modelpath /home/ubuntu/data/mica-character-attribution/70-finetune/sft-results/2000-trope-2025May15-174725 \
--model sft \
--dataset chatter-contexts-original-semantic-2000-dev \
--dataset chatter-contexts-original-semantic-2000-test \
--prediction_batch_size 4
# --dataset personet-dev \
# --dataset personet-test

# # run inference on chatter-segments using SFT-trained on chatter-contexts-first-2000
# $CMD --modelpath /home/ubuntu/data/mica-character-attribution/70-finetune/sft-results/2000-first-2025May16-042313 \
# --model sft \
# --dataset chatter-segments-original-test \
# --dataset chatter-segments-anonymized-test \
# --prediction_batch_size 1

# # run inference on chatter-contexts-first-2000 & personet using SFT-trained on chatter-contexts-first-2000
# $CMD --modelpath /home/ubuntu/data/mica-character-attribution/70-finetune/sft-results/2000-first-2025May16-042313 \
# --model sft \
# --dataset chatter-contexts-original-first-2000-dev \
# --dataset chatter-contexts-original-first-2000-test \
# --dataset personet-dev \
# --dataset personet-test \
# --prediction_batch_size 4