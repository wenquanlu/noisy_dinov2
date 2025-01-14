for step in 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss100-resume-0-140-200-0-60-60/config.yaml \
        --pretrained-weights output_gauss100-resume-0-140-200-0-60-60/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss100-resume-0-140-200-0-60-60/eval/training_$step/linear_epoch30 \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)
done