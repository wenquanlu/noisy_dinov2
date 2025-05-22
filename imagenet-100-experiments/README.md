## ImageNet-100 Experiments
All experiments are conducted within imagenet-100-experiments/ directory.
```shell
cd imagenet-100-experiments
```
### Create Noisy Dataset
First download [ImageNet-100](https://drive.google.com/file/d/1gBbVGzQxXXUe9HMClEdvCmIEPPj1Y8i1/view?usp=sharing) (100 classes: 50k training, 5k validation)
```shell
python utils/generate_noise.py
```

### Linear Probing Evaluation
Train Neighbor2Neighbor and Denoise
```shell
mv noisy_mini-imagenet* Neighbor2Neighbor/
cd Neighbor2Neighbor
sh linear_probe_n2n.sh
```

Train and evaluate DINOv2, N2N + DINOv2, DINOv2 w/ NC, DINOv2 w/ NCT for all noise types presented, make sure you are at imagenet-100-experiments/ directory
```shell
mv Neighbor2Neighbor/noisy_mini-imagenet* .
python process_metadata.py

# train and evaluate DINOv2, N2N + DINOv2, DINOv2 w/ NC
sh exps/linear_probe.sh

# train and evaluate DINOv2 w/ NCT (need to get denoised weights first)
sh exps/linear_probe_nct.sh
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
# this will generate instance_recog.sh
python exps/generate_instance_recog_script.py
sh instance_recog.sh
```