#!/bin/bash

#SBATCH -n 4
#SBATCH -c 6
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH -t 1:00:00
#SBATCH -p YOUR_PARTITION


PYTHONPATH="PATH_TO_imagenet-1k-experiments" python dinov2/run/train/train.py \
    --nodes 1 \
    --ngpus 4 \
    --config-file dinov2/configs/imagenet1k_gauss255_vitb_150.yaml \
    --output-dir output_gauss255-noisy-150 \
    train.dataset_path=ImageNet:split=TRAIN:root=imagenet-gauss255:extra=imagenet-gauss255-extra
