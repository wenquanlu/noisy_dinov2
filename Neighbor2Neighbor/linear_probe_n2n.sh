## prepare (preprocess) training set
python dataset_tool.py --input_dir="noisy_mini-imagenet-gauss50/train" --save_dir="Imagenet_train_gauss50"
python dataset_tool.py --input_dir="noisy_mini-imagenet-gauss100/train" --save_dir="Imagenet_train_gauss100"
python dataset_tool.py --input_dir="noisy_mini-imagenet-gauss255/train" --save_dir="Imagenet_train_gauss255"
python dataset_tool.py --input_dir="noisy_mini-imagenet-speckle0.4/train" --save_dir="Imagenet_train_speckle0.4"
python dataset_tool.py --input_dir="noisy_mini-imagenet-speckle0.7/train" --save_dir="Imagenet_train_speckle0.7"
python dataset_tool.py --input_dir="noisy_mini-imagenet-speckle1.0/train" --save_dir="Imagenet_train_speckle1.0"
python dataset_tool.py --input_dir="noisy_mini-imagenet-shot10/train" --save_dir="Imagenet_train_shot10"
python dataset_tool.py --input_dir="noisy_mini-imagenet-shot3/train" --save_dir="Imagenet_train_shot3"
python dataset_tool.py --input_dir="noisy_mini-imagenet-shot1/train" --save_dir="Imagenet_train_shot1"


## train neighbor2neighbor
for noise in gauss50 gauss100 gauss255 speckle0.4 speckle0.7 speckle1.0 shot10 shot3 shot1
do
    python train.py \
    --data_dir=./Imagenet_train_$noise \
    --val_dirs=./validation \
    --noisetype=$noise \
    --save_model_path=./results \
    --log_name=unet_${noise}_epoch100 \
    --increase_ratio=2
done

## set up denoised repository
for noise in gauss50 gauss100 gauss255 speckle0.4 speckle0.7 speckle1.0 shot10 shot3 shot1
do
    python testdata_structure.py $noise
done


## convert noisy imagenet to denoised
python eval.py --state_dict=TRAINED_WEIGHTS_PATH --data_dir=./noisy_mini-imagenet-gauss50 --save_dir=./noisy_mini-imagenet-gauss50-denoised

python eval.py --state_dict=TRAINED_WEIGHTS_PATH --data_dir=./noisy_mini-imagenet-gauss100 --save_dir=./noisy_mini-imagenet-gauss100-denoised

python eval.py --state_dict=TRAINED_WEIGHTS_PATH --data_dir=./noisy_mini-imagenet-gauss255 --save_dir=./noisy_mini-imagenet-gauss255-denoised

python eval.py --state_dict=TRAINED_WEIGHTS_PATH --data_dir=./noisy_mini-imagenet-speckle0.4 --save_dir=./noisy_mini-imagenet-speckle0.4-denoised

python eval.py --state_dict=TRAINED_WEIGHTS_PATH --data_dir=./noisy_mini-imagenet-speckle0.7 --save_dir=./noisy_mini-imagenet-speckle0.7-denoised

python eval.py --state_dict=TRAINED_WEIGHTS_PATH --data_dir=./noisy_mini-imagenet-speckle1.0 --save_dir=./noisy_mini-imagenet-speckle1.0-denoised

python eval.py --state_dict=TRAINED_WEIGHTS_PATH --data_dir=./noisy_mini-imagenet-shot10 --save_dir=./noisy_mini-imagenet-shot10-denoised

python eval.py --state_dict=TRAINED_WEIGHTS_PATH --data_dir=./noisy_mini-imagenet-shot3 --save_dir=./noisy_mini-imagenet-shot3-denoised

python eval.py --state_dict=TRAINED_WEIGHTS_PATH --data_dir=./noisy_mini-imagenet-shot1 --save_dir=./noisy_mini-imagenet-shot1-denoised