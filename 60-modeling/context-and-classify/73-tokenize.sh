#!/bin/bash
MODELS=(llama)
SOURCES=(all utter desc)
CONTEXTS=(0 1 5 10 20)
LOGLENS=(13 14 15 16)

for MODEL in "${MODELS[@]}"; do
    for SOURCE in "${SOURCES[@]}"; do
        for CONTEXT in "${CONTEXTS[@]}"; do
            for LOGLEN in "${LOGLENS[@]}"; do
                echo $MODEL $SOURCE $CONTEXT $LOGLEN
                python 70-classification/73-tokenize.py --model=$MODEL --source=$SOURCE --context=$CONTEXT \
                --loglen=$LOGLEN
            done
        done
    done
done