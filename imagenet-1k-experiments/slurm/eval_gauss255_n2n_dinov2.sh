#!/bin/bash

#SBATCH -n 4
#SBATCH -c 6
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH -t 1:00:00
#SBATCH -p YOUR_PARTITION

PYTHONPATH="PATH_TO_imagenet-1k-experiments" python dinov2/run/eval/linear.py \
    --nodes 1 \
    --ngpus 4 \
    --batch-size 256 \
    --num-workers 6 \
    --epochs 20 \
    --pretrained-weights output_gauss255-denoised/eval/training_249999/teacher_checkpoint.pth \
    --config-file output_gauss255-denoised/config.yaml \
    --output-dir output_gauss255-denoised/eval/training_249999/linear \
    --train-dataset ImageNet:split=TRAIN:root=imagenet-gauss255-denoised:extra=imagenet-gauss255-denoised-extra \
    --val-dataset ImageNet:split=VAL:root=imagenet-gauss255-denoised:extra=imagenet-gauss255-denoised-extra
