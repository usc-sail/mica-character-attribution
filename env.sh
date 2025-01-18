#!/bin/bash

conda create -n chatter python=3.12

conda install -c conda-forge numpy pandas tqdm selenium absl-py requests beautifulsoup4 unidecode scikit-learn scipy ipykernel matplotlib spacy cupy thefuzz nameparser jupyterlab ipython cinemagoer openai tiktoken tenacity boto3 sqlite gpustat htop google-generativeai google-cloud-aiplatform ipywidgets datasets
pip install krippendorff pylcs

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install conda-forge::transformers
pip install accelerate
pip install bitsandbytes
pip install sentencepiece

pip install pytorch torchvision torchaudio
absl-py
pandas
transformers
accelerate
deepspeed
bitsandbytes
sentencepiece
google-generativeai
google-cloud-aiplatform
scikit-learn
jupyterlab
ipywidgets