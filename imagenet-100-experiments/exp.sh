JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/imagenet_10_200_mixed_config.yaml \
    --output-dir output_imagenet10-50-gauss01-teachernoise \
    --mix_train_single_noise \
    --mix_train_std 0.4 \
    train.dataset_path=ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra)

# JOB_ID=$(python dinov2/run/eval/linear.py \
#         --config-file output_imagenet10-50-gauss01-single-noise-global/config.yaml \
#         --pretrained-weights output_imagenet10-50-gauss01-single-noise-global/eval/training_24999/teacher_checkpoint.pth \
#         --output-dir output_imagenet10-50-gauss01-single-noise-global/eval/training_24999/linear \
#         --train-dataset ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra \
#         --val-dataset ImageNet:split=VAL:root=imagenet-10-50:extra=imagenet-10-50-extra)

# JOB_ID=$(python dinov2/run/train/train.py \
#     --nodes 1 \
#     --config-file dinov2/configs/imagenet_10_200_mixed_config.yaml \
#     --output-dir output_imagenet10-50-gauss04-baseline \
#     train.dataset_path=ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra)

# JOB_ID=$(python dinov2/run/eval/linear.py \
#         --config-file output_imagenet10-50-gauss04-baseline-02/config.yaml \
#         --pretrained-weights output_imagenet10-50-gauss04-baseline-02/eval/training_24999/teacher_checkpoint.pth \
#         --output-dir output_imagenet10-50-gauss04-baseline-02/eval/training_24999/linear \
#         --train-dataset ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra \
#         --val-dataset ImageNet:split=VAL:root=imagenet-10-50:extra=imagenet-10-50-extra)
