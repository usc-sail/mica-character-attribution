#!/bin/bash
# Hyperparameter Tuning

## Learning Rate
# LRS=(1e-5 2e-5 5e-5 1e-4)
# for LR in "${LRS[@]}"; do
#     accelerate launch --config_file deepspeed_config.yaml 512-finetune.py \
#         --data_file contexts/25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl \
#         --max_seq_len 1800 \
#         --model meta-llama/Llama-3.1-8B-Instruct \
#         --load_4bit \
#         --flash_attn \
#         --train_batch_size 2 \
#         --eval_batch_size 8 \
#         --eval_on_start \
#         --lr $LR \
#         --noalso_eval_devset \
#         --train_steps 512 \
#         --eval_steps 32 \
#         --logtofile \
#         --save_predictions
# done

## LoRA Rank
# RANKS=(128 64 32 16 8)
# for RANK in "${RANKS[@]}"; do
#     accelerate launch --config_file deepspeed_config.yaml 512-finetune.py \
#         --data_file contexts/25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl \
#         --max_seq_len 1800 \
#         --model meta-llama/Llama-3.1-8B-Instruct \
#         --load_4bit \
#         --flash_attn \
#         --train_batch_size 2 \
#         --eval_batch_size 8 \
#         --eval_on_start \
#         --lr 2e-5 \
#         --rank $RANK \
#         --alpha $RANK \
#         --noalso_eval_devset \
#         --train_steps 512 \
#         --eval_steps 32 \
#         --logtofile \
#         --save_predictions
# done

# # LoRA Alpha
# ALPHAS=(64 16)
# for ALPHA in "${ALPHAS[@]}"; do
#     accelerate launch --config_file deepspeed_config.yaml 512-finetune.py \
#         --data_file contexts/25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl \
#         --max_seq_len 1800 \
#         --model meta-llama/Llama-3.1-8B-Instruct \
#         --load_4bit \
#         --flash_attn \
#         --train_batch_size 2 \
#         --eval_batch_size 8 \
#         --eval_on_start \
#         --lr 2e-5 \
#         --rank 32 \
#         --alpha $ALPHA \
#         --noalso_eval_devset \
#         --train_steps 512 \
#         --eval_steps 32 \
#         --logtofile \
#         --save_predictions
# done

# Different sizes for strategy=trope neg-sim=-0.05 pos-sim=0.05
CMD="accelerate launch --config_file deepspeed_config.yaml 512-finetune.py \
     --model meta-llama/Llama-3.1-8B-Instruct --load_4bit --flash_attn \
     --lr 2e-5 --rank 32 --alpha 64 --logtofile --eval_on_start --save_predictions"
WORDS=(250 500 1000 1500 2000)
TOKENS=(500 1000 1800 2700 3500)
TRAIN_BATCH_SIZES=(8 4 2 1 1)
EVAL_BATCH_SIZES=(32 16 8 4 4)
MAX_STEPS=(256 512 1024 2048 2048)
EVAL_STEPS=(8 16 32 64 64)
SIMS=(0.05 0.1 0.15 0.2)
# for i in {0..4}; do
#     $CMD \
#     --data_file contexts/25P-"${WORDS[i]}"C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl --max_seq_len "${TOKENS[i]}" \
#     --train_batch_size "${TRAIN_BATCH_SIZES[i]}" --eval_batch_size "${EVAL_BATCH_SIZES[i]}" \
#     --eval_steps "${EVAL_STEPS[i]}" --train_steps "${MAX_STEPS[i]}"
# done

# # Strategy and Size
# STRATEGIES=("random" "first" "last")
# MAX_STEPS=(128 256 512 1024 1024)
# for STRATEGY in "${STRATEGIES[@]}"; do
#     for i in {0..4}; do
#         $CMD \
#         --data_file contexts/25P-"${WORDS[i]}"C-$STRATEGY.jsonl --max_seq_len "${TOKENS[i]}" \
#         --train_batch_size "${TRAIN_BATCH_SIZES[i]}" --eval_batch_size "${EVAL_BATCH_SIZES[i]}" \
#         --eval_steps "${EVAL_STEPS[i]}" --train_steps "${MAX_STEPS[i]}"
#     done
# done

# # Similarities
# for SIM in "${SIMS[@]}"; do
#     $CMD \
#     --data_file contexts/25P-1000C-all-mpnet-base-v2-"$SIM"NEG-"$SIM"POS.jsonl --max_seq_len 1800 \
#     --train_batch_size 2 --eval_batch_size 8 --eval_steps 32 --train_steps 512
# done

# BEST MODEL
$CMD --data_file contexts/25P-1000C-all-mpnet-base-v2-0.05NEG-0.05POS.jsonl --max_seq_len 1800 \
    --train_batch_size 2 --eval_batch_size 8 --eval_steps 32 --train_steps 512 --save_model --also_eval_devset