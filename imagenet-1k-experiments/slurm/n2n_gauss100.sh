#!/bin/bash

#SBATCH -n 24
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH -t 24:00:00
#SBATCH -p YOUR_PARTITION

python -u train.py \
    --data_dir=./imagenet_train_gauss100_subset \
    --val_dirs=./validation \
    --noisetype=gauss100 \
    --save_model_path=./results \
    --log_name=unet_gauss100_epoch100 \
    --increase_ratio=2 \
    --batchsize=8 > log.txt
