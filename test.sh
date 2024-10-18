python dinov2/run/eval/linear.py \
    --config-file output/config.yaml \
    --pretrained-weights output/eval/training_24999/teacher_checkpoint.pth \
    --output-dir output/eval/training_24999/linear \
    --train-dataset ImageNet:split=TRAIN:root=imagenet-100:extra=imagenet-100-extra \
    --val-dataset ImageNet:split=VAL:root=imagenet-100:extra=imagenet-100-extra