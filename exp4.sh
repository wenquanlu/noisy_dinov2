for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss255-resume-0-140-200-0-60-60-repeat/config.yaml \
        --pretrained-weights output_gauss255-resume-0-140-200-0-60-60-repeat/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss255-resume-0-140-200-0-60-60-repeat/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)
done