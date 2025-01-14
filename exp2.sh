# for step in 74999 68749 62499 56249 6249 12499 18749 24999 31249 37499 43749 49999 
# do
#     JOB_ID=$(python dinov2/run/eval/linear.py \
#         --config-file output_gauss255-resume-0-140-200-0-60-60-reg-ibot/config.yaml \
#         --pretrained-weights output_gauss255-resume-0-140-200-0-60-60-reg-ibot/eval/training_$step/teacher_checkpoint.pth \
#         --output-dir output_gauss255-resume-0-140-200-0-60-60-reg-ibot/eval/training_$step/linear \
#         --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra \
#         --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)
# done


for step in 74999 68749 62499 56249 6249 12499 18749 24999 31249 37499 43749 49999 
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot1-resume-0-140-200-0-60-60-all-02/config.yaml \
        --pretrained-weights output_shot1-resume-0-140-200-0-60-60-all-02/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot1-resume-0-140-200-0-60-60-all-02/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)
done