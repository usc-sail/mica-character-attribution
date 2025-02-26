#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --time=2-00:00:00
#SBATCH --account=shrikann_35

source ~/.bashrc
cd /home1/sbaruah/mica-character-attribution
source .venv/bin/activate
cd 50-modeling
accelerate launch --config_file deepspeed.yaml --num_processes 2 --gpu_ids 0,1 \
    509-extracts.py --partition dev --slice $1 --nslices $2 \
    --hf_model meta-llama/Llama-3.1-8B-Instruct --attn flash_attention_2 \
    --max_input_tokens 64 --max_output_tokens 1536 --do_sample --top_p 0.9