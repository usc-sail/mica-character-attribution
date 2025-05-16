#!/bin/bash

# CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
# accelerate launch --config_file deepspeed.yaml 509-finetune.py \
#     --contexts_file 25P-1000C-first.jsonl \
#     --model meta-llama/Llama-3.1-8B --attn flash_attention_2 --load_4bit \
#     --train_batch_size 8 --train_steps 2048 \
#     --eval_batch_size 32 --eval_delay 1800 --eval_steps 32 --logging_steps 2 --save_model \
#     --lr 2e-5 --warmup_steps 0 \
#     --rank 32 --alpha 64 \
#     --lora_target_module q_proj --lora_target_module k_proj --lora_target_module v_proj \
#     --alsologtostderr --noshowprefixforinfo

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1

CMD="accelerate launch --config_file deepspeed.yaml 509-finetune.py \
--model meta-llama/Llama-3.1-8B --attn flash_attention_2 --load_4bit \
--train_batch_size 8 --train_steps 2048 \
--noeval --eval_batch_size 32 --logging_steps 32 --save_model \
--lr 2e-5 --warmup_steps 0 \
--rank 32 --alpha 64 \
--lora_target_module q_proj --lora_target_module k_proj --lora_target_module v_proj \
--alsologtostderr --noshowprefixforinfo"

# $CMD --contexts_file 25P-500C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl
# $CMD --contexts_file 25P-250C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl

# $CMD --contexts_file 25P-2000C-first.jsonl
# $CMD --contexts_file 25P-2000C-last.jsonl
# $CMD --contexts_file 25P-2000C-random.jsonl

# for SIZE in 2000 1000 1500 500 250; do
#     $CMD --contexts_file 25P-${SIZE}C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl
#     for STRATEGY in first last random; do
#         $CMD --contexts_file 25P-${SIZE}C-${STRATEGY}.jsonl
#     done
# done