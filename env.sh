#!/bin/bash

conda create -n attr python=3.11

conda install -c conda-forge numpy pandas tqdm selenium absl-py requests beautifulsoup4 unidecode scikit-learn scipy ipykernel matplotlib spacy cupy thefuzz nameparser jupyterlab ipython cinemagoer

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::transformers