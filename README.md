# Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum

Self-Supervised Learning (SSL) has become a powerful solution to extract rich representations from unlabeled data. Yet, SSL research is mostly focused on clean, curated and high-quality datasets. As a result, applying SSL on noisy data remains a challenge, despite being crucial to applications such as astrophysics, medical imaging, geophysics or finance. In this work, we present a fully self-supervised framework that enables noise-robust representation learning without requiring a denoiser at inference or downstream fine-tuning. Our method first trains an SSL denoiser on noisy data, then uses it to construct a denoised-to-noisy data curriculum (i.e., training first on denoised, then noisy samples) for pretraining a SSL backbone (e.g., DINOv2), combined with a teacher-guided regularization that anchors noisy embeddings to their denoised counterparts. This process encourages the model to internalize noise robustness. Notably, the denoiser can be discarded after pretraining, simplifying deployment. On ImageNet-1k with ViT-B under extreme Gaussian noise ($\sigma=255$, SNR = 0.72 dB), our method improves linear probing accuracy by 4.8\% over DINOv2, demonstrating that denoiser-free robustness can emerge from noise-aware pretraining.

<figure>
<img src="img/noise_grid_long_figu_2.jpg">
<!--<img src="img/noisy_framework.png">
<img src="img/dinov2_regularization.png">-->
</figure>

## Get Started

### Create Noisy Dataset
First download [ImageNet-100](https://drive.google.com/file/d/1gBbVGzQxXXUe9HMClEdvCmIEPPj1Y8i1/view?usp=sharing) (100 classes: 50k training, 5k validation)
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
python process_metadata.py
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
## Comments
Our codebase builds heavily on [DINOv2](https://github.com/facebookresearch/dinov2) and [Neighbor2Neighbor](https://github.com/TaoHuang2018/Neighbor2Neighbor).
