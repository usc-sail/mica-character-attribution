#!/bin/bash
# chatter word size = instruction token size : 250 = 550, 500 = 1000, 1000 = 1800, 1500 = 2700, 2000 = 3500

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1

# main training command
CMD="accelerate launch --config_file deepspeed.yaml 509-finetune.py --instrtune \
--model meta-llama/Llama-3.1-8B-Instruct --attn flash_attention_2 --load_4bit --train_batch_size 1 \
--noeval --eval_batch_size 4 --logging_steps 32 --save_model \
--lr 2e-5 --warmup_steps 0 \
--rank 32 --alpha 64 \
--lora_target_module q_proj --lora_target_module k_proj --lora_target_module v_proj \
--alsologtostderr --noshowprefixforinfo"

# instruct-tune on chatter and test on chatter & personet
CHATTER_CMD="${CMD} --train_dataset chatter --train_steps 1024"
PERSONET_CMD="${CMD} --train_dataset personet --personet_instr_seqlen 1700 --train_steps 512"

OLDIFS=$IFS
IFS=","
for SIZE in 2000,3500 1000,1800 1500,2700 500,1000 250,550; do
    set -- $SIZE
    $CHATTER_CMD --contexts_file 25P-${1}C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl --chatter_instr_seqlen $2
    for STRATEGY in first last random; do
        $CHATTER_CMD --contexts_file 25P-${1}C-${STRATEGY}.jsonl --chatter_instr_seqlen $2
    done
done
IFS=$OLDIFS

# instruct-tune on personet and test on chatter and personet
$PERSONET_CMD --contexts_file 25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl --chatter_instr_seqlen 1800