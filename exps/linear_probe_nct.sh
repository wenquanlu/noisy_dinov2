JOB_ID=$(DENOISED_CKPT="output_gauss255-200-denoised/model_0174999.rank" NOISE_TYPE="gauss255" REG_STRENGTH="1.1" python dinov2_reg/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_gauss255_140_config.yaml \
    --output-dir output_gauss255-resume-0-140-200-0-60-60-all-11 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)


for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
            --config-file output_gauss255-resume-0-140-200-0-60-60-all-11/config.yaml \
            --pretrained-weights output_gauss255-resume-0-140-200-0-60-60-all-11/eval/training_$step/teacher_checkpoint.pth \
            --output-dir output_gauss255-resume-0-140-200-0-60-60-all-11/eval/training_$step/linear \
            --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra \
            --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)
done


JOB_ID=$(DENOISED_CKPT="output_gauss100-200-denoised/model_0174999.rank" NOISE_TYPE="gauss100" REG_STRENGTH="4.0" python dinov2_reg/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_140_config.yaml \
    --output-dir output_gauss100-resume-0-140-200-0-60-60-all-40 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)


for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss100-resume-0-140-200-0-60-60-all-40/config.yaml \
        --pretrained-weights output_gauss100-resume-0-140-200-0-60-60-all-40/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss100-resume-0-140-200-0-60-60-all-40/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)
done


JOB_ID=$(DENOISED_CKPT="output_shot1-200-denoised/model_0174999.rank" NOISE_TYPE="shot1" REG_STRENGTH="5.0" python dinov2_reg/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_shot1_140_config.yaml \
    --output-dir output_shot1-resume-0-140-200-0-60-60-all-50 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)


for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot1-resume-0-140-200-0-60-60-all-50/config.yaml \
        --pretrained-weights output_shot1-resume-0-140-200-0-60-60-all-50/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot1-resume-0-140-200-0-60-60-all-50/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)
done


JOB_ID=$(DENOISED_CKPT="output_speckle1.0-200-denoised/model_0174999.rank" NOISE_TYPE="speckle1.0" REG_STRENGTH="2.0" python dinov2_reg/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_speckle1.0_140_config.yaml \
    --output-dir output_speckle1.0-resume-0-140-200-0-60-60-all-20 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle1.0-resume-0-140-200-0-60-60-all-20/config.yaml \
        --pretrained-weights output_speckle1.0-resume-0-140-200-0-60-60-all-20/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle1.0-resume-0-140-200-0-60-60-all-20/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra)
done
