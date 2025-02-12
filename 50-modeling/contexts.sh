#!/bin/bash

# MIN_POS_SIMS=(0.05 0.1 0.15 0.2)
# MAX_NEG_SIMS=(-0.2 -0.1 -0.15 -0.05)
# COUNTER=0
# for MIN_POS_SIM in "${MIN_POS_SIMS[@]}"; do
#     for MAX_NEG_SIM in "${MAX_NEG_SIMS[@]}"; do
#         COUNTER=$(((COUNTER+1)%8))
#         CUDA_VISIBLE_DEVICES=$COUNTER python 506-contexts.py \
#             --max_words_per_context 1000 \
#             --strategy trope \
#             --min_pos_similarity $MIN_POS_SIM \
#             --max_neg_similarity $MAX_NEG_SIM &
#     done
# done
SIMS=(0.05 0.1 0.15 0.2)
WORDS=(250 500 1500 2000)
COUNTER=0
for SIM in "${SIMS[@]}"; do
    for WORD in "${WORDS[@]}"; do
        CUDA_VISIBLE_DEVICES=$COUNTER python 506-contexts.py \
            --max_words_per_context $WORD \
            --strategy trope \
            --min_pos_similarity $SIM \
            --max_neg_similarity -$SIM &
        COUNTER=$(((COUNTER+1)%8))
    done
done