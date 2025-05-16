#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --time=2-00:00:00
#SBATCH --account=shrikann_35

source ~/.bashrc
cd /home1/sbaruah/mica-character-attribution
source .venv/bin/activate
cd 50-modeling
CMD="accelerate launch --config_file deepspeed.yaml --num_processes 2 --gpu_ids 0,1 508-extracts.py --partition dev \
    --hf_model meta-llama/Llama-3.1-8B-Instruct --attn flash_attention_2 \
    --max_input_tokens 64 --max_output_tokens 1536 --do_sample --top_p 0.9"
# $CMD --slice $(($1)) --nslices $2
# $CMD --slice $(($1 + 1)) --nslices $2
# $CMD --slice $(($1 + 2)) --nslices $2
# $CMD --slice $(($1 + 3)) --nslices $2
$CMD --slice 14 --nslices 16
$CMD --slice 15 --nslices 16