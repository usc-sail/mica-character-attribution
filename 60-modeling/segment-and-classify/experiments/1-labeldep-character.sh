#!/bin/bash

# CUDA_VISIBLE_DEVICES=$1 python 60-modeling/64-train.py \
#     --lbl \
#     --segment --context=10 \
#     --model=longformer --longformerattn=128 \
#     --tokbatch=40 \
#     --mention \
#     --utter \
#     --ep=5 \
#     --alsologtostderr

# CUDA_VISIBLE_DEVICES=$1 python 60-modeling/64-train.py \
#     --lbl \
#     --segment --context=10 \
#     --model=longformer --longformerattn=512 \
#     --nolora \
#     --tokbatch=40 \
#     --mention \
#     --utter \
#     --ep=5 \
#     --trpbatch=200 \
#     --batcheval=1000 \
#     --elr=1e-4 \
#     --alsologtostderr

CUDA_VISIBLE_DEVICES=$1 python 60-modeling/64-train.py \
    --lbl \
    --segment --context=10 \
    --model=llama\
    --lora \
    --tokbatch=32 \
    --mention \
    --utter \
    --ep=5 \
    --trpbatch=200 \
    --elr=1e-4 \
    --nosave_logs \
    --alsologtostderr