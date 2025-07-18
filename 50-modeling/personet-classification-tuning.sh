#!/bin/bash

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
accelerate launch --config_file deepspeed.yaml 509-finetune.py --train_dataset personet \
    --contexts_file 25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl \
    --model meta-llama/Llama-3.1-8B --attn flash_attention_2 --load_4bit \
    --train_batch_size 8 --train_steps 1024 \
    --eval_batch_size 32 --noeval --logging_steps 32 --save_model \
    --lr 2e-5 --warmup_steps 0 \
    --rank 32 --alpha 64 \
    --lora_target_module q_proj --lora_target_module k_proj --lora_target_module v_proj \
    --alsologtostderr --noshowprefixforinfo