#!/bin/bash

accelerate launch --config_file="70-classification/deepspeed-config.yaml" --multi-gpu \
    70-classification/74-train-ddp.py --source=all --context=0 --loglen=14 --train_batch_size=1 --optim=adamw_torch_4bit