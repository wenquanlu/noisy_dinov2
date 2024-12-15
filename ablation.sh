# train 140 (gauss100)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_140_norestart_config.yaml \
    --output-dir output_gauss100-resume-0-140-200-0-60-60-norestart \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss100-resume-0-140-200-0-60-60-norestart/config.yaml \
        --pretrained-weights output_gauss100-resume-0-140-200-0-60-60-norestart/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss100-resume-0-140-200-0-60-60-norestart/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)
done



JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_noisy_config.yaml \
    --output-dir output_gauss100-200-save \
    --max_to_keep=45 \
    --save_frequency=5 \
   train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)


# train 140 (gauss100)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_noisy_resume_140_config.yaml \
    --output-dir output_gauss100-noisy-resume-0-140-200-0-60-60 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss100-noisy-resume-0-140-200-0-60-60/config.yaml \
        --pretrained-weights output_gauss100-noisy-resume-0-140-200-0-60-60/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss100-noisy-resume-0-140-200-0-60-60/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)
done