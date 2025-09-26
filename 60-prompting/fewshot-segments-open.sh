#!/bin/bash
accelerate launch --config_file deepspeed.yaml --num_processes 2 505-fewshot-segments.py --runs 3 \
    --shots 2 --hf_model meta-llama/Llama-3.1-8B-Instruct --attn flash_attention_2 --max_output_tokens 1
accelerate launch --config_file deepspeed.yaml --num_processes 2 505-fewshot-segments.py --runs 3 \
    --shots 2 --hf_model mistralai/Mistral-Nemo-Instruct-2407 --attn flash_attention_2 --max_output_tokens 1
accelerate launch --config_file deepspeed.yaml --num_processes 2 505-fewshot-segments.py --runs 3 \
    --shots 2 --hf_model microsoft/Phi-3-small-128k-instruct --max_output_tokens 1