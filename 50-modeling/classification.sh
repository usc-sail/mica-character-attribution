#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1

# main training command
CMD="accelerate launch --config_file deepspeed.yaml 509-finetune.py \
--model meta-llama/Llama-3.1-8B --attn flash_attention_2 --load_4bit \
--train_batch_size 8 \
--noeval --eval_batch_size 32 --logging_steps 32 --save_model \
--lr 2e-5 --warmup_steps 0 \
--rank 32 --alpha 64 \
--lora_target_module q_proj --lora_target_module k_proj --lora_target_module v_proj \
--alsologtostderr --noshowprefixforinfo"

CHATTER_CMD="${CMD} --train_dataset chatter --train_steps 2048"
PERSONET_CMD="${CMD} --train_dataset personet --train_steps 1024"

# train classifier on chatter and test on chatter & personet
for SIZE in 2000 1000 1500 500 250; do
    echo $CHATTER_CMD --contexts_file 25P-${SIZE}C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl
    for STRATEGY in first last random; do
        echo $CHATTER_CMD --contexts_file 25P-${SIZE}C-${STRATEGY}.jsonl
    done
done

# train classifier on personet and test on chatter & personet
echo $PERSONET_CMD --contexts_file 25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl