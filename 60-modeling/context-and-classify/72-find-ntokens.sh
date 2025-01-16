#!/bin/bash
MODELS=(llama)
SOURCES=(all utter desc)
CONTEXTS=(0 1 5 10 20)

for MODEL in "${MODELS[@]}"; do
    for SOURCE in "${SOURCES[@]}"; do
        for CONTEXT in "${CONTEXTS[@]}"; do
            echo $MODEL $SOURCE $CONTEXT
            python 70-classification/72-find-ntokens.py --model=$MODEL --source=$SOURCE --context=$CONTEXT
        done
    done
done