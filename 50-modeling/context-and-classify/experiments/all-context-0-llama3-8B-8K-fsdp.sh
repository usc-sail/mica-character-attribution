#!/bin/bash

accelerate launch --config_file="70-classification/fsdp-config.yaml" --multi-gpu \
    70-classification/74-train-ddp.py --source=all --context=0 --loglen=13 --train_batch_size=4