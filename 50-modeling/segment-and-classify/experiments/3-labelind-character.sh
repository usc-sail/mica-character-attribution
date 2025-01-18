#!/bin/bash
# CUDA_VISIBLE_DEVICES=$1 python 60-modeling/64-train.py \
#     --nolbl \
#     --segment --context=0 \
#     --model=roberta \
#     --tokbatch=80 \
#     --mention \
#     --utter \
#     --alsologtostderr





CUDA_VISIBLE_DEVICES=$1 python 60-modeling/64-train.py \
    --nolbl \
    --segment --context=0 \
    --model=longformer \
    --longformerattn=128 \
    --tokbatch=40 \
    --mention \
    --utter \
    --alsologtostderr