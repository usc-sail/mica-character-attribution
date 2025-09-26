#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1

CLS="accelerate launch --config_file deepspeed.yaml 509-finetune.py \
--contexts_file 25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl \
--attn flash_attention_2 --load_4bit \
--train_batch_size 8 --train_steps 2048 \
--noeval --eval_batch_size 32 --logging_steps 32 --save_model \
--lr 2e-5 --warmup_steps 0 \
--rank 32 --alpha 64 \
--lora_target_module q_proj --lora_target_module k_proj --lora_target_module v_proj \
--alsologtostderr --noshowprefixforinfo"

INS="accelerate launch --config_file deepspeed.yaml 509-finetune.py --instrtune \
--contexts_file 25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl --max_seq_len 1800 \
--attn flash_attention_2 --load_4bit \
--train_batch_size 1 --train_steps 1024 \
--noeval --eval_batch_size 4 --logging_steps 32 \
--lr 2e-5 --warmup_steps 0 \
--rank 32 --alpha 64 \
--lora_target_module q_proj --lora_target_module k_proj --lora_target_module v_proj \
--alsologtostderr --noshowprefixforinfo"

$CLS --model google/gemma-2-9b
$INS --model google/gemma-2-9b-it