#!/bin/bash

#SBATCH -n 4
#SBATCH -c 6
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH -t 00:10:00
#SBATCH -p YOUR_PARTITION

PYTHONPATH="PATH_TO_imagenet-1k-experiments" python dinov2/run/train/train.py \
    --nodes 1 \
    --ngpus 4 \
    --config-file dinov2/configs/imagenet1k_gauss100_vitb_restart30_70_warmup15.yaml \
    --output-dir output_gauss100-restart30_70_warmup15 \
    train.dataset_path=ImageNet:split=TRAIN:root=imagenet-gauss100:extra=imagenet-gauss100-extra
