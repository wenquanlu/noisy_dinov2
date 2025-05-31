JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/imagenet_10_200_mixed_config.yaml \
    --output-dir output_imagenet10-50-gauss04-baseline-02 \
    train.dataset_path=ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra)


if [ -d "output_imagenet10-50-gauss04-baseline-02/eval/training_24999" ]; then
        echo "Training output found. Proceeding to evaluation..."
else
        rm -rf output_imagenet10-50-gauss04-baseline-02
        JOB_ID=$(python dinov2/run/train/train.py \
            --nodes 1 \
            --config-file dinov2/configs/imagenet_10_200_mixed_config.yaml \
            --output-dir output_imagenet10-50-gauss04-baseline-02 \
            train.dataset_path=ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra)
fi

if [ -d "output_imagenet10-50-gauss04-baseline-02/eval/training_24999" ]; then
        echo "Training output found. Proceeding to evaluation..."
else
        rm -rf output_imagenet10-50-gauss04-baseline-02
        JOB_ID=$(python dinov2/run/train/train.py \
            --nodes 1 \
            --config-file dinov2/configs/imagenet_10_200_mixed_config.yaml \
            --output-dir output_imagenet10-50-gauss04-baseline-02 \
            train.dataset_path=ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra)
fi


if [ -d "output_imagenet10-50-gauss04-baseline-02/eval/training_24999" ]; then
        echo "Training output found. Proceeding to evaluation..."
else
        rm -rf output_imagenet10-50-gauss04-baseline-02
        JOB_ID=$(python dinov2/run/train/train.py \
            --nodes 1 \
            --config-file dinov2/configs/imagenet_10_200_mixed_config.yaml \
            --output-dir output_imagenet10-50-gauss04-baseline-02 \
            train.dataset_path=ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra)
fi



if [ -d "output_imagenet10-50-gauss04-baseline-02/eval/training_24999" ]; then
        echo "Training output found. Proceeding to evaluation..."
else
        rm -rf output_imagenet10-50-gauss04-baseline-02
        JOB_ID=$(python dinov2/run/train/train.py \
            --nodes 1 \
            --config-file dinov2/configs/imagenet_10_200_mixed_config.yaml \
            --output-dir output_imagenet10-50-gauss04-baseline-02 \
            train.dataset_path=ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra)
fi



if [ -d "output_imagenet10-50-gauss04-baseline-02/eval/training_24999" ]; then
        echo "Training output found. Proceeding to evaluation..."
else
        rm -rf output_imagenet10-50-gauss04-baseline-02
        JOB_ID=$(python dinov2/run/train/train.py \
            --nodes 1 \
            --config-file dinov2/configs/imagenet_10_200_mixed_config.yaml \
            --output-dir output_imagenet10-50-gauss04-baseline-02 \
            train.dataset_path=ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra)
fi



if [ -d "output_imagenet10-50-gauss04-baseline-02/eval/training_24999" ]; then
        echo "Training output found. Proceeding to evaluation..."
else
        rm -rf output_imagenet10-50-gauss04-baseline-02
        JOB_ID=$(python dinov2/run/train/train.py \
            --nodes 1 \
            --config-file dinov2/configs/imagenet_10_200_mixed_config.yaml \
            --output-dir output_imagenet10-50-gauss04-baseline-02 \
            train.dataset_path=ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra)
fi



if [ -d "output_imagenet10-50-gauss04-baseline-02/eval/training_24999" ]; then
        echo "Training output found. Proceeding to evaluation..."
else
        rm -rf output_imagenet10-50-gauss04-baseline-02
        JOB_ID=$(python dinov2/run/train/train.py \
            --nodes 1 \
            --config-file dinov2/configs/imagenet_10_200_mixed_config.yaml \
            --output-dir output_imagenet10-50-gauss04-baseline-02 \
            train.dataset_path=ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra)
fi


if [ -d "output_imagenet10-50-gauss04-baseline-02/eval/training_24999" ]; then
        echo "Training output found. Proceeding to evaluation..."
else
        rm -rf output_imagenet10-50-gauss04-baseline-02
        JOB_ID=$(python dinov2/run/train/train.py \
            --nodes 1 \
            --config-file dinov2/configs/imagenet_10_200_mixed_config.yaml \
            --output-dir output_imagenet10-50-gauss04-baseline-02 \
            train.dataset_path=ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra)
fi

if [ -d "output_imagenet10-50-gauss04-baseline-02/eval/training_24999" ]; then
        echo "Training output found. Proceeding to evaluation..."
else
        rm -rf output_imagenet10-50-gauss04-baseline-02
        JOB_ID=$(python dinov2/run/train/train.py \
            --nodes 1 \
            --config-file dinov2/configs/imagenet_10_200_mixed_config.yaml \
            --output-dir output_imagenet10-50-gauss04-baseline-02 \
            train.dataset_path=ImageNet:split=TRAIN:root=imagenet-10-50:extra=imagenet-10-50-extra)
fi


# if [ -d "output_gauss255-resume-0-500-500-0-500-500-regularized-11/eval/training_624999" ]; then
#         echo "Training output found. Proceeding to evaluation..."
# else
#         rm -rf output_gauss255-resume-0-500-500-0-500-500-regularized-11
#         JOB_ID=$(DENOISED_CKPT="output_gauss255-500-denoised/model_0624999.rank" NOISE_TYPE="gauss255" REG_STRENGTH="1.1" python dinov2_reg/run/train/train.py \
#             --nodes 1 \
#             --config-file dinov2/configs/gauss255_1000_restart500.yaml \
#             --output-dir output_gauss255-resume-0-500-500-0-500-500-regularized-11 \
#             train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)
# fi
