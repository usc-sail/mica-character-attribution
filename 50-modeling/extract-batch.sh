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
CMD="python 509-extracts.py --partition train --nslices $2 --llama_model Llama-3.1-8B-Instruct --flash_attn \
 --batch_size 1 --max_input_tokens 64 --max_output_tokens 3584 --temperature 1 --stream"
$CMD --slice $1 --device cuda:0 &
$CMD --slice $(($1+1)) --device cuda:1