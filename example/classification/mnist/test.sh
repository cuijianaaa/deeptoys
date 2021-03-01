#!/bin/bash

export PYTHONUNBUFFERED="True"

GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID \
python ../../../model/classification/mnist/pipe.py \
      --cfg cfg.yml \
      --mode test \
      --cuda | tee results/val.log
