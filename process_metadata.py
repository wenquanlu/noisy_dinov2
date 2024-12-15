from dinov2.data.datasets import ImageNet

noises = [
    "gauss50", "gauss100", "gauss255", "speckle0.4", "speckle0.7", "speckle1.0", "shot10", "shot3", "shot1"
]

for noise in noises:
    for split in ImageNet.Split:
        noisy_dataset = ImageNet(split=split, root=f"noisy_mini-imagenet-{noise}", extra="noisy_mini-imagenet-{noise}-extra")
        noisy_dataset.dump_extra()
        denoised_dataset = ImageNet(split=split, root=f"noisy_mini-imagenet-{noise}-denoised", extra="noisy_mini-imagenet-{noise}-denoised-extra")
        denoised_dataset.dump_extra()