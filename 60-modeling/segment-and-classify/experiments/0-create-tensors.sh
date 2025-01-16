#!/bin/bash
for MODEL in roberta longformer; do
    python 60-modeling/61-save-story-tensors.py --model=$MODEL --alsologtostderr
    for CONTEXT in 0 5 10; do
        python 60-modeling/62-save-character-tensors.py --model=$MODEL --context=$CONTEXT --alsologtostderr
    done
done