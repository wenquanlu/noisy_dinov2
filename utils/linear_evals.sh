#!/bin/bash

# Iterate over each weight and run the evaluation command
for weight in training_12499 training_49999 training_74999 training_87499
do
    echo "Running evaluation for weight: $weight"
    python dinov2/run/eval/linear.py \
        --config-file output/config.yaml \
        --pretrained-weights output/eval/$weight/teacher_checkpoint.pth \
        --output-dir output/eval/$weight/linear \
        --train-dataset ImageNet:split=TRAIN:root=imagenet-100:extra=imagenet-100-extra \
        --val-dataset ImageNet:split=VAL:root=imagenet-100:extra=imagenet-100-extra
    sleep 1700
done


# training_112499 training_12499 training_124999 training_37499 training_49999 training_62499 training_74999 training_87499 training_99999

##
python dinov2/run/eval/linear.py \
--config-file output/config.yaml \
--pretrained-weights output/eval/training_49999/teacher_checkpoint.pth \
--output-dir output/eval/training_49999/linear \
--train-dataset ImageNet:split=TRAIN:root=imagenet-100:extra=imagenet-100-extra \
--val-dataset ImageNet:split=VAL:root=imagenet-100:extra=imagenet-100-extra