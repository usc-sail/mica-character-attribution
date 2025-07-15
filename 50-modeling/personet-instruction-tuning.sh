#!/bin/bash

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
accelerate launch --config_file deepspeed.yaml 509-finetune.py --instrtune \
    --train_dataset personet --personet_instr_seqlen 1700 \
    --contexts_file 25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl --chatter_instr_seqlen 1800 \
    --model meta-llama/Llama-3.1-8B-Instruct --attn flash_attention_2 --load_4bit \
    --train_batch_size 1 --train_steps 512 \
    --eval_batch_size 4 --noeval --eval_delay 0 --eval_steps 32 --logging_steps 8 --save_model \
    --lr 2e-5 --warmup_steps 0 \
    --rank 32 --alpha 64 \
    --lora_target_module q_proj --lora_target_module k_proj --lora_target_module v_proj \
    --alsologtostderr --noshowprefixforinfo