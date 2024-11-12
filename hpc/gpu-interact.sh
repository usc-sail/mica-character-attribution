#!/bin/bash
# v100(32)/p100(16)/a40(48)/a100(40)
# add --constraint=a100-80gb to specifically ask for a100 80 GB gpu

if [ -z "$1" ]
then
    salloc \
    --account=shrikann_35 \
    --time=2-00:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --partition=gpu \
    --gres=gpu:a100:1 \
    --constraint=a100-80gb
else
    salloc \
    --account=shrikann_35 \
    --time=2-00:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --partition=gpu \
    --gres=gpu:$1:1
fi