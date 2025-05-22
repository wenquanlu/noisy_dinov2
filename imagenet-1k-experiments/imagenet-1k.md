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
For efficiency, we use a 100k subset of ImageNet-1k to train Neighbor2Neighbor
```shell
python dataset_tool_subset.py --input_dir=imagenet-gauss100/train/ --save_dir=imagenet_train_gauss100_subset

python dataset_tool_subset.py --input_dir=imagenet-gauss255/train/ --save_dir=imagenet_train_gauss255_subset
```
Train Neighbor2Neighbor:
```shell
sbatch n2n_gauss100.sh
sbatch n2n_gauss255.sh
```
Use eval.py to denoise the dataset. If you have multiple gpus, you can partition the imagenet to speed up the process.
```shell
python -u eval.py --state_dict=<STATE_DICT> --data_dir=<NOISY_DIR> --save_dir=<DENOISED_DIR>
```
In addition, we also provide our pretrained Neighbor2Neighbor weights: 
| Gauss100 Denoiser | Guass255 Denoiser |
|-------------------|-------------------|
|[Download](https://drive.google.com/file/d/1bggz_pVl24FKkvPqmfEAxjV0MXdRe8NE/view?usp=sharing) | [Download](https://drive.google.com/file/d/1uhImyaEbumC4FfdzKJ5SEYyfjeX9ZhJy/view?usp=sharing) |


### Train DINOv2 and Linear Evaluation

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