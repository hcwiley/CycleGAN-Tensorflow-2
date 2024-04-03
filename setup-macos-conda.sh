#!/bin/bash

# helper script to setup conda environment for cyclegan on macOS

conda create -n cyclegan python=3.10

conda activate cyclegan

conda install -c apple scikit-image tqdm tensorflow=2.9

conda install -c conda-forge oyaml

pip install tensorflow-addons==0.17.1 tensorflow-macos==2.9.0 tensorflow-metal==0.5.0