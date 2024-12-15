# From Denoised to Noisy: A Fully Self-supervised Framework for Robust Representation Learning on Noisy Image Data

In this work, we present a fully self-supervised framework designed to enable DINOv2 to learn noise-robust representations from only noisy image data. The proposed method first trains a self-supervised denoiser to bootstrap a denoised dataset from the noisy dataset. We then train DINOv2 on the denoised images, followed by resetting the training dynamics and restarting the training on the original noisy images. This curriculum approach substantially improves classification and instance recognition performance of DINOv2 compared to training solely on noisy images. Remarkably, the performance closely matches or even surpasses that of models trained exclusively on denoised images. By enabling DINOv2 to adapt independently to noise, our method introduces an effective paradigm for self-supervised learning with noisy data.

<figure>
<img src="img/noisy_arch.png">
</figure>

## Get Started
Please download our curated mini-imagenet using this [link](https://drive.google.com/file/d/1kUbAt-FST_ptL-i-rxN46I-8P0xCDF3g/view?usp=sharing), and put it in current folder.

### Create Noisy Dataset
```shell
python utils/generate_noise.py
```

### Linear Probing
Train Neighbor2Neighbor and Denoise
```shell
mv noisy_mini-imagenet* Neighbor2Neighbor/
cd Neighbor2Neighbor
sh linear_probe_n2n.sh
```

Train and Evaluate DINOv2, make sure you are at outer directory
```shell
mv Neighbor2Neighbor/noisy_mini-imagenet* .
sh linear_probe.sh
```

### Instance Recognition
Download Oxford and Paris dataset from the official site

Use Neighbor2Neighbor to denoise
```shell
cd Neighbor2Neighbor
sh instance_recog_n2n.sh
mv roxford* OXFORD_PARIS_DATASET_PATH/data/datasets/
mv rparis* OXFORD_PARIS_DATASET_PATH/data/datasets/
```

Evaluate DINOv2, make sure you are at outer directory
```shell
python generate_instance_recog_script.py
sh instance_recog.sh
```

