from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="mini-imagenet", extra="mini-imagenet-extra")
    dataset.dump_extra()