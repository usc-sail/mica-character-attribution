#!/bin/bash
# 250 = 550, 500 = 1000, 1000 = 1800, 1500 = 2700, 2000 = 3500

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
accelerate launch --config_file deepspeed.yaml 509-finetune.py --instrtune --train_dataset chatter \
    --contexts_file 25P-2000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl --chatter_instr_seqlen 3500 \
    --model meta-llama/Llama-3.1-8B-Instruct --attn flash_attention_2 --load_4bit \
    --train_batch_size 1 --train_steps 512 \
    --eval_batch_size 4 --eval_delay 512 --eval_steps 32 --logging_steps 8 \
    --lr 2e-5 --warmup_steps 0 \
    --rank 32 --alpha 64 \
    --lora_target_module q_proj --lora_target_module k_proj --lora_target_module v_proj \
    --alsologtostderr --noshowprefixforinfo