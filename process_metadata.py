from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="imagenet-100", extra="imagenet-100-extra")
    dataset.dump_extra()