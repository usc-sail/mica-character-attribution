#!/bin/bash

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
accelerate launch --config_file deepspeed.yaml 509-finetune.py \
--contexts_file 25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl \
--model meta-llama/Llama-3.1-8B --attn flash_attention_2 --load_4bit \
--train_batch_size 16 --train_steps 1024 \
--eval_batch_size 64 --eval_delay 256 --eval_steps 32 --logging_steps 8 \
--lr 2e-5 \
--alsologtostderr --noshowprefixforinfo