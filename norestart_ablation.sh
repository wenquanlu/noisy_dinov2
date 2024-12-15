# # train 140 (gauss100)
# JOB_ID=$(python dinov2/run/train/train.py \
#     --nodes 1 \
#     --config-file dinov2/configs/ssl_resume_140_norestart_config.yaml \
#     --output-dir output_gauss100-resume-0-140-200-0-60-60-norestart \
#     train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)

for step in 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss100-resume-0-140-200-0-60-60-norestart/config.yaml \
        --pretrained-weights output_gauss100-resume-0-140-200-0-60-60-norestart/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss100-resume-0-140-200-0-60-60-norestart/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)
done
