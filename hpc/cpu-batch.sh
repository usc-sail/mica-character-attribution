#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=epyc-64
#SBATCH --time=2-00:00:00
#SBATCH --account=shrikann_35

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate /home1/sbaruah/.conda/envs/coreference
cd /home1/sbaruah/mica_text_coref

bash experiments/baselines/lea.sh $1