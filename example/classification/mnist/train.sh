#!/bin/bash

export PYTHONUNBUFFERED="True"

GPU_ID=$1

mkdir results

CUDA_VISIBLE_DEVICES=$GPU_ID \
python ../../../model/classification/mnist/pipe.py \
      --cfg cfg.yml \
      --mode train \
      --cuda | tee results/train.log
