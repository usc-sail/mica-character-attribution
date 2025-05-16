#!/bin/bash
accelerate launch --config_file deepspeed.yaml --num_processes 2 504-zeroshot-segments.py --anonymize \
    --hf_model microsoft/Phi-3-small-128k-instruct --attn eager \
    --max_input_tokens 16 --max_output_tokens 256
accelerate launch --config_file deepspeed.yaml --num_processes 2 504-zeroshot-segments.py --anonymize \
    --hf_model mistralai/Mistral-Nemo-Instruct-2407 --attn flash_attention_2 \
    --max_input_tokens 16 --max_output_tokens 256
accelerate launch --config_file deepspeed.yaml --num_processes 2 504-zeroshot-segments.py --anonymize \
    --hf_model meta-llama/Llama-3.1-8B-Instruct --attn flash_attention_2 \
    --max_input_tokens 16 --max_output_tokens 256