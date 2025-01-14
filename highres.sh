# JOB_ID=$(python dinov2/run/train/train.py \
#     --nodes 1 \
#     --config-file dinov2/configs/high_res_clean_config.yaml \
#     --output-dir output_clean-200-highres-2 \
#     --img_format=".JPEG" \
#    train.dataset_path=ImageNet:split=TRAIN:root=mini-imagenet:extra=mini-imagenet-extra)

for step in 39999 34999 29999 24999 4999 9999 14999 19999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_clean-200-highres-2/config.yaml \
        --pretrained-weights output_clean-200-highres-2/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_clean-200-highres-2/eval/training_$step/linear \
        --img_format=".JPEG" \
        --train-dataset ImageNet:split=TRAIN:root=mini-imagenet:extra=mini-imagenet-extra \
        --val-dataset ImageNet:split=VAL:root=mini-imagenet:extra=mini-imagenet-extra)
done