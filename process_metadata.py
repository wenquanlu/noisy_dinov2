from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="noisy_mini-imagenet-speckle1.0-denoised", extra="noisy_mini-imagenet-speckle1.0-denoised-extra")
    dataset.dump_extra()