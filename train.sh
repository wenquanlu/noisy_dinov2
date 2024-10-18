python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_default_config.yaml \
    --output-dir output \
    train.dataset_path=ImageNet:split=TRAIN:root=imagenet-100:extra=imagenet-100-extra