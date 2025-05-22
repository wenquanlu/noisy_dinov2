## ImageNet-1k Experiments
All experiments are conducted within imagenet-1k-experiments/ directory. We use slurm to launch experiments.
```shell
cd imagenet-1k-experiments
```
### Create Noisy Dataset
First download and unzip imagenet-1k, and structure the dataset as follows
```shell
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

Then run the command below. The command uses multiple threads to speed up the noise addition process
```shell
python utils/generate_noise_multi.py
```

### Train Neighbor2Neighbor Denoiser





### Training and Linear Evaluation

```shell
# dinov2 (gauss100)
sbatch slurm/gauss100_dinov2.sh
sbatch slurm/eval_gauss100_dinov2.sh

# n2n + dinov2 (gauss100)
sbatch slurm/gauss100_n2n_dinov2.sh
sbatch slurm/eval_gauss100_n2n_dinov2.sh

# dinov2 w/ NC (gauss100)
sbatch slurm/gauss100_dinov2_w_nc.sh
sbatch slurm/eval_gauss100_dinov2_w_nc.sh

# dinov2 w/ NCT (gauss100)
sbatch slurm/gauss100_dinov2_w_nct.sh
sbatch slurm/eval_gauss100_dinov2_w_nct.sh

# dinov2 (gauss255)
sbatch slurm/gauss255_dinov2.sh
sbatch slurm/eval_gauss255_dinov2.sh

# n2n + dinov2 (gauss255)
sbatch slurm/gauss255_n2n_dinov2.sh
sbatch slurm/eval_gauss255_n2n_dinov2.sh

# dinov2 w/ NC (gauss255)
sbatch slurm/gauss255_dinov2_w_nc.sh
sbatch slurm/eval_gauss255_dinov2_w_nc.sh

# dinov2 w/ NCT (gauss255)
sbatch slurm/gauss255_dinov2_w_nct.sh
sbatch slurm/eval_gauss255_dinov2_w_nct.sh
```