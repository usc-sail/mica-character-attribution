#!/bin/bash

conda create -n chatter python=3.12

conda install -c conda-forge numpy pandas tqdm selenium absl-py requests beautifulsoup4 unidecode scikit-learn scipy ipykernel matplotlib spacy cupy thefuzz nameparser jupyterlab ipython cinemagoer openai tiktoken tenacity boto3 sqlite gpustat htop

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install conda-forge::transformers