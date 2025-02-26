#!/bin/bash

accelerate launch --config_file deepspeed.yaml --num_processes 2 512-finetune.py \
    --contexts_file 25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl \
    --model meta-llama/Llama-3.1-8B --attn flash_attention_2 --load_4bit \
    --train_batch_size 16 --eval_batch_size 64 \
    --train_steps 48 --eval_steps 32 --logging_steps 8 \
    --logtofile --alsologtostderr --noshowprefixforinfo