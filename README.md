# Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum

Self-Supervised Learning (SSL) has become a powerful solution to extract rich representations from unlabeled data. Yet, SSL research is mostly focused on clean, curated and high-quality datasets. As a result, applying SSL on noisy data remains a challenge, despite being crucial to applications such as astrophysics, medical imaging, geophysics or finance. In this work, we present a fully self-supervised framework that enables noise-robust representation learning without requiring a denoiser at inference or downstream fine-tuning. Our method first trains an SSL denoiser on noisy data, then uses it to construct a denoised-to-noisy data curriculum (i.e., training first on denoised, then noisy samples) for pretraining a SSL backbone (e.g., DINOv2), combined with a teacher-guided regularization that anchors noisy embeddings to their denoised counterparts. This process encourages the model to internalize noise robustness. Notably, the denoiser can be discarded after pretraining, simplifying deployment. On ImageNet-1k with ViT-B under extreme Gaussian noise ($\sigma=255$, SNR = 0.72 dB), our method improves linear probing accuracy by 4.8\% over DINOv2, demonstrating that denoiser-free robustness can emerge from noise-aware pretraining.

<figure>
<img src="img/noise_grid_long_figu_2.jpg">
<!--<img src="img/noisy_framework.png">
<img src="img/dinov2_regularization.png">-->
</figure>

## MLP Toy Experiment on MNIST
Please refer to [toy_exp.md](toy_mnist/toy_exp.md) for running toy experiments.


## ImageNet-100 Experiments
Please refer to [imagenet-100.md](imagenet-100-experiments/imagenet-100.md) for running ImageNet-100 experiments.

## ImageNet-1k Experiments
Please refer to [imagenet-1k.md](imagenet-1k-experiments/imagenet-1k.md) for running ImageNet-1k experiments.


## Acknowledgement
Our codebase builds heavily on [DINOv2](https://github.com/facebookresearch/dinov2) and [Neighbor2Neighbor](https://github.com/TaoHuang2018/Neighbor2Neighbor).


If you find this repo helpful, please consider giving this repo a star :star:.
