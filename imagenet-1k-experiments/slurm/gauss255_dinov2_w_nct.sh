#!/bin/bash

#SBATCH -n 4
#SBATCH -c 6
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH -t 00:10:00
#SBATCH -p YOUR_PARTITION

PYTHONPATH="PATH_TO_imagenet-1k-experiments" DENOISED_CKPT="output_gauss255-denoised/model_0249999.rank" NOISE_TYPE="gauss255" REG_STRENGTH="1.6" python dinov2_reg/run/train/train.py \
    --nodes 1 \
    --ngpus 4 \
    --config-file dinov2/configs/imagenet1k_gauss255_vitb_restart30_70.yaml \
    --output-dir output_gauss255-regularized30_70_16 \
    train.dataset_path=ImageNet:split=TRAIN:root=imagenet-gauss255:extra=imagenet-gauss255-extra
