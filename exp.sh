# # train 140 (gauss255)
# JOB_ID=$(python dinov2/run/train/train.py \
#     --nodes 1 \
#     --config-file dinov2/configs/ssl_resume_gauss255_140_config.yaml \
#     --output-dir output_gauss255-resume-0-140-200-0-60-60-reg-ibot \
#     train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)

# for step in 56249 62499 68749 74999 6249 12499 18749 24999 31249 37499 43749 49999 
# do
#     JOB_ID=$(python dinov2/run/eval/linear.py \
#         --config-file output_gauss255-resume-0-140-200-0-60-60-reg/config.yaml \
#         --pretrained-weights output_gauss255-resume-0-140-200-0-60-60-reg/eval/training_$step/teacher_checkpoint.pth \
#         --output-dir output_gauss255-resume-0-140-200-0-60-60-reg/eval/training_$step/linear \
#         --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra \
#         --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)
# done

# python dinov2/run/eval/linear.py \
#         --config-file output_gauss255-resume-0-140-200-0-60-60-reg/config.yaml \
#         --pretrained-weights output_gauss255-200-denoised/eval/training_174999/teacher_checkpoint.pth \
#         --output-dir output_gauss255-resume-0-140-200-0-60-60-reg/eval/training_0/linear \
#         --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra \
#         --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra

# train 140 (gauss255)
# JOB_ID=$(python dinov2_reg/run/train/train.py \
#     --nodes 1 \
#     --config-file dinov2/configs/ssl_resume_gauss255_140_config.yaml \
#     --output-dir output_gauss255-resume-0-140-200-0-60-60-all-04 \
#     train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)

# for step in 74999 68749 62499 56249 6249 12499 18749 24999 31249 37499 43749 49999 
# do
#     JOB_ID=$(python dinov2/run/eval/linear.py \
#         --config-file output_gauss255-resume-0-140-200-0-60-60-all-04/config.yaml \
#         --pretrained-weights output_gauss255-resume-0-140-200-0-60-60-all-04/eval/training_$step/teacher_checkpoint.pth \
#         --output-dir output_gauss255-resume-0-140-200-0-60-60-all-04/eval/training_$step/linear \
#         --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra \
#         --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)
# done

JOB_ID=$(python dinov2_reg/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_shot1_140_config.yaml \
    --output-dir output_shot1-resume-0-140-200-0-60-60-all-03 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)

for step in 74999 68749 62499 56249 6249 12499 18749 24999 31249 37499 43749 49999 
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot1-resume-0-140-200-0-60-60-all-03/config.yaml \
        --pretrained-weights output_shot1-resume-0-140-200-0-60-60-all-03/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot1-resume-0-140-200-0-60-60-all-03/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)
done

# train 140 (gauss100)
JOB_ID=$(python dinov2_reg/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_gauss100_140_config.yaml \
    --output-dir output_gauss100-resume-0-140-200-0-60-60-all-03 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)

for step in 74999 68749 62499 56249 6249 12499 18749 24999 31249 37499 43749 49999 
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss100-resume-0-140-200-0-60-60-all-03/config.yaml \
        --pretrained-weights output_gauss100-resume-0-140-200-0-60-60-all-03/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss100-resume-0-140-200-0-60-60-all-03/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)
done