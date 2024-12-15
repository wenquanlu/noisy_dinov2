### noisy image (gauss100) 8h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_noisy_config.yaml \
    --output-dir output_gauss100-200 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss100-200/config.yaml \
        --pretrained-weights output_gauss100-200/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss100-200/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)
done

### denoised image (gauss100) 8h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_denoised_config.yaml \
    --output-dir output_gauss100-200-denoised \
    --max_to_keep=45 \
    --save_frequency=5 \
   train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100-denoised:extra=noisy_mini-imagenet-gauss100-denoised-extra)


### evaluate denoised image (gauss100) 14h
for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss100-200-denoised/config.yaml \
        --pretrained-weights output_gauss100-200-denoised/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss100-200-denoised/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100-denoised:extra=noisy_mini-imagenet-gauss100-denoised-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss100-denoised:extra=noisy_mini-imagenet-gauss100-denoised-extra)
done

# train 130 (gauss100)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_130_config.yaml \
    --output-dir output_gauss100-resume-0-130-200-0-70-70 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)


for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss100-resume-0-130-200-0-70-70/config.yaml \
        --pretrained-weights output_gauss100-resume-0-130-200-0-70-70/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss100-resume-0-130-200-0-70-70/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)
done

# train 140 (gauss100)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_140_config.yaml \
    --output-dir output_gauss100-resume-0-140-200-0-60-60 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss100-resume-0-140-200-0-60-60/config.yaml \
        --pretrained-weights output_gauss100-resume-0-140-200-0-60-60/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss100-resume-0-140-200-0-60-60/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)
done


# train 120 (gauss100)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_120_config.yaml \
    --output-dir output_gauss100-resume-0-120-200-0-80-80 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss100-resume-0-120-200-0-80-80/config.yaml \
        --pretrained-weights output_gauss100-resume-0-120-200-0-80-80/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss100-resume-0-120-200-0-80-80/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss100:extra=noisy_mini-imagenet-gauss100-extra)
done

###################################################################################
### noisy image (gauss50) 8h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_noisy_config.yaml \
    --output-dir output_gauss50-200 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss50:extra=noisy_mini-imagenet-gauss50-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss50-200/config.yaml \
        --pretrained-weights output_gauss50-200/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss50-200/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss50:extra=noisy_mini-imagenet-gauss50-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss50:extra=noisy_mini-imagenet-gauss50-extra)
done

### denoised image (gauss50) 8h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_denoised_config.yaml \
    --output-dir output_gauss50-200-denoised \
    --max_to_keep=45 \
    --save_frequency=5 \
   train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss50-denoised:extra=noisy_mini-imagenet-gauss50-denoised-extra)


### evaluate denoised image (gauss50) 14h
for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss50-200-denoised/config.yaml \
        --pretrained-weights output_gauss50-200-denoised/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss50-200-denoised/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss50-denoised:extra=noisy_mini-imagenet-gauss50-denoised-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss50-denoised:extra=noisy_mini-imagenet-gauss50-denoised-extra)
done

# train 130 (gauss50)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_gauss50_130_config.yaml \
    --output-dir output_gauss50-resume-0-130-200-0-70-70 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss50:extra=noisy_mini-imagenet-gauss50-extra)


for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss50-resume-0-130-200-0-70-70/config.yaml \
        --pretrained-weights output_gauss50-resume-0-130-200-0-70-70/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss50-resume-0-130-200-0-70-70/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss50:extra=noisy_mini-imagenet-gauss50-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss50:extra=noisy_mini-imagenet-gauss50-extra)
done

# train 140 (gauss50)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_gauss50_140_config.yaml \
    --output-dir output_gauss50-resume-0-140-200-0-60-60 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss50:extra=noisy_mini-imagenet-gauss50-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss50-resume-0-140-200-0-60-60/config.yaml \
        --pretrained-weights output_gauss50-resume-0-140-200-0-60-60/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss50-resume-0-140-200-0-60-60/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss50:extra=noisy_mini-imagenet-gauss50-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss50:extra=noisy_mini-imagenet-gauss50-extra)
done


# train 120 (gauss50)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_gauss50_120_config.yaml \
    --output-dir output_gauss50-resume-0-120-200-0-80-80 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss50:extra=noisy_mini-imagenet-gauss50-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss50-resume-0-120-200-0-80-80/config.yaml \
        --pretrained-weights output_gauss50-resume-0-120-200-0-80-80/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss50-resume-0-120-200-0-80-80/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss50:extra=noisy_mini-imagenet-gauss50-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss50:extra=noisy_mini-imagenet-gauss50-extra)
done

#######################################################################################################
### noisy image (gauss255) 22h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_noisy_config.yaml \
    --output-dir output_gauss255-200 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss255-200/config.yaml \
        --pretrained-weights output_gauss255-200/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss255-200/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)
done


### denoised image (gauss255) 8h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_denoised_config.yaml \
    --output-dir output_gauss255-200-denoised \
    --max_to_keep=45 \
    --save_frequency=5 \
   train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255-denoised:extra=noisy_mini-imagenet-gauss255-denoised-extra)


### evaluate denoised image (gauss255) 14h
for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss255-200-denoised/config.yaml \
        --pretrained-weights output_gauss255-200-denoised/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss255-200-denoised/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255-denoised:extra=noisy_mini-imagenet-gauss255-denoised-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss255-denoised:extra=noisy_mini-imagenet-gauss255-denoised-extra)
done

# train 130 (gauss255)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_gauss255_130_config.yaml \
    --output-dir output_gauss255-resume-0-130-200-0-70-70 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)


for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss255-resume-0-130-200-0-70-70/config.yaml \
        --pretrained-weights output_gauss255-resume-0-130-200-0-70-70/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss255-resume-0-130-200-0-70-70/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)
done

# train 140 (gauss255)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_gauss255_140_config.yaml \
    --output-dir output_gauss255-resume-0-140-200-0-60-60 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss255-resume-0-140-200-0-60-60/config.yaml \
        --pretrained-weights output_gauss255-resume-0-140-200-0-60-60/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss255-resume-0-140-200-0-60-60/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)
done


# train 120 (gauss255)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_gauss255_120_config.yaml \
    --output-dir output_gauss255-resume-0-120-200-0-80-80 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_gauss255-resume-0-120-200-0-80-80/config.yaml \
        --pretrained-weights output_gauss255-resume-0-120-200-0-80-80/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_gauss255-resume-0-120-200-0-80-80/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-gauss255:extra=noisy_mini-imagenet-gauss255-extra)
done

############################################################################
### noisy imaghe (speckle0.7) 22h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_noisy_config.yaml \
    --output-dir output_speckle0.7-200 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.7:extra=noisy_mini-imagenet-speckle0.7-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle0.7-200/config.yaml \
        --pretrained-weights output_speckle0.7-200/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle0.7-200/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.7:extra=noisy_mini-imagenet-speckle0.7-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle0.7:extra=noisy_mini-imagenet-speckle0.7-extra)
done

### denoised image (speckle0.7) 8h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_denoised_config.yaml \
    --output-dir output_speckle0.7-200-denoised \
    --max_to_keep=45 \
    --save_frequency=5 \
   train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.7-denoised:extra=noisy_mini-imagenet-speckle0.7-denoised-extra)



### evaluate denoised image (speckle 0.7) 14h
for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle0.7-200-denoised/config.yaml \
        --pretrained-weights output_speckle0.7-200-denoised/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle0.7-200-denoised/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.7-denoised:extra=noisy_mini-imagenet-speckle0.7-denoised-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle0.7-denoised:extra=noisy_mini-imagenet-speckle0.7-denoised-extra)
done

# train 130 (speckle0.7)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_speckle0.7_130_config.yaml \
    --output-dir output_speckle0.7-resume-0-130-200-0-70-70 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.7:extra=noisy_mini-imagenet-speckle0.7-extra)


for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle0.7-resume-0-130-200-0-70-70/config.yaml \
        --pretrained-weights output_speckle0.7-resume-0-130-200-0-70-70/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle0.7-resume-0-130-200-0-70-70/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.7:extra=noisy_mini-imagenet-speckle0.7-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle0.7:extra=noisy_mini-imagenet-speckle0.7-extra)
done

# train 140 (speckle0.7)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_speckle0.7_140_config.yaml \
    --output-dir output_speckle0.7-resume-0-140-200-0-60-60 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.7:extra=noisy_mini-imagenet-speckle0.7-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle0.7-resume-0-140-200-0-60-60/config.yaml \
        --pretrained-weights output_speckle0.7-resume-0-140-200-0-60-60/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle0.7-resume-0-140-200-0-60-60/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.7:extra=noisy_mini-imagenet-speckle0.7-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle0.7:extra=noisy_mini-imagenet-speckle0.7-extra)
done


# train 120 (speckle0.7)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_speckle0.7_120_config.yaml \
    --output-dir output_speckle0.7-resume-0-120-200-0-80-80 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.7:extra=noisy_mini-imagenet-speckle0.7-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle0.7-resume-0-120-200-0-80-80/config.yaml \
        --pretrained-weights output_speckle0.7-resume-0-120-200-0-80-80/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle0.7-resume-0-120-200-0-80-80/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.7:extra=noisy_mini-imagenet-speckle0.7-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle0.7:extra=noisy_mini-imagenet-speckle0.7-extra)
done

############################################################################################################


### noisy image (speckle1.0) 22h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_noisy_config.yaml \
    --output-dir output_speckle1.0-200 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle1.0-200/config.yaml \
        --pretrained-weights output_speckle1.0-200/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle1.0-200/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra)
done


### denoised image (speckle1.0) 8h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_denoised_config.yaml \
    --output-dir output_speckle1.0-200-denoised \
    --max_to_keep=45 \
    --save_frequency=5 \
   train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle1.0-denoised:extra=noisy_mini-imagenet-speckle1.0-denoised-extra)


### evaluate denoised image (speckle1.0) 14h
for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle1.0-200-denoised/config.yaml \
        --pretrained-weights output_speckle1.0-200-denoised/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle1.0-200-denoised/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle1.0-denoised:extra=noisy_mini-imagenet-speckle1.0-denoised-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle1.0-denoised:extra=noisy_mini-imagenet-speckle1.0-denoised-extra)
done

# train 130 (speckle1.0)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_speckle1.0_130_config.yaml \
    --output-dir output_speckle1.0-resume-0-130-200-0-70-70 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra)


for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle1.0-resume-0-130-200-0-70-70/config.yaml \
        --pretrained-weights output_speckle1.0-resume-0-130-200-0-70-70/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle1.0-resume-0-130-200-0-70-70/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra)
done

# train 140 (speckle1.0)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_speckle1.0_140_config.yaml \
    --output-dir output_speckle1.0-resume-0-140-200-0-60-60 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle1.0-resume-0-140-200-0-60-60/config.yaml \
        --pretrained-weights output_speckle1.0-resume-0-140-200-0-60-60/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle1.0-resume-0-140-200-0-60-60/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra)
done


# train 120 (speckle1.0)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_speckle1.0_120_config.yaml \
    --output-dir output_speckle1.0-resume-0-120-200-0-80-80 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle1.0-resume-0-120-200-0-80-80/config.yaml \
        --pretrained-weights output_speckle1.0-resume-0-120-200-0-80-80/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle1.0-resume-0-120-200-0-80-80/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle1.0:extra=noisy_mini-imagenet-speckle1.0-extra)
done

############################################################################################
### noisy imaghe (speckle0.4) 22h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_noisy_config.yaml \
    --output-dir output_speckle0.4-200 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.4:extra=noisy_mini-imagenet-speckle0.4-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle0.4-200/config.yaml \
        --pretrained-weights output_speckle0.4-200/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle0.4-200/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.4:extra=noisy_mini-imagenet-speckle0.4-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle0.4:extra=noisy_mini-imagenet-speckle0.4-extra)
done


### denoised image (speckle0.4) 8h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_denoised_config.yaml \
    --output-dir output_speckle0.4-200-denoised \
    --max_to_keep=45 \
    --save_frequency=5 \
   train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.4-denoised:extra=noisy_mini-imagenet-speckle0.4-denoised-extra)



### evaluate denoised image (speckle 0.4) 14h
for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle0.4-200-denoised/config.yaml \
        --pretrained-weights output_speckle0.4-200-denoised/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle0.4-200-denoised/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.4-denoised:extra=noisy_mini-imagenet-speckle0.4-denoised-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle0.4-denoised:extra=noisy_mini-imagenet-speckle0.4-denoised-extra)
done

# train 130 (speckle0.4)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_speckle0.4_130_config.yaml \
    --output-dir output_speckle0.4-resume-0-130-200-0-70-70 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.4:extra=noisy_mini-imagenet-speckle0.4-extra)


for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle0.4-resume-0-130-200-0-70-70/config.yaml \
        --pretrained-weights output_speckle0.4-resume-0-130-200-0-70-70/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle0.4-resume-0-130-200-0-70-70/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.4:extra=noisy_mini-imagenet-speckle0.4-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle0.4:extra=noisy_mini-imagenet-speckle0.4-extra)
done

# train 140 (speckle0.4)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_speckle0.4_140_config.yaml \
    --output-dir output_speckle0.4-resume-0-140-200-0-60-60 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.4:extra=noisy_mini-imagenet-speckle0.4-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle0.4-resume-0-140-200-0-60-60/config.yaml \
        --pretrained-weights output_speckle0.4-resume-0-140-200-0-60-60/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle0.4-resume-0-140-200-0-60-60/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.4:extra=noisy_mini-imagenet-speckle0.4-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle0.4:extra=noisy_mini-imagenet-speckle0.4-extra)
done


# train 120 (speckle0.4)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_speckle0.4_120_config.yaml \
    --output-dir output_speckle0.4-resume-0-120-200-0-80-80 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.4:extra=noisy_mini-imagenet-speckle0.4-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_speckle0.4-resume-0-120-200-0-80-80/config.yaml \
        --pretrained-weights output_speckle0.4-resume-0-120-200-0-80-80/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_speckle0.4-resume-0-120-200-0-80-80/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-speckle0.4:extra=noisy_mini-imagenet-speckle0.4-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-speckle0.4:extra=noisy_mini-imagenet-speckle0.4-extra)
done

#################################################################################################

### noisy shot3 22h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_noisy_config.yaml \
    --output-dir output_shot3-200 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot3:extra=noisy_mini-imagenet-shot3-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot3-200/config.yaml \
        --pretrained-weights output_shot3-200/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot3-200/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot3:extra=noisy_mini-imagenet-shot3-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot3:extra=noisy_mini-imagenet-shot3-extra)
done


### denoised image (shot3) 8h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_denoised_config.yaml \
    --output-dir output_shot3-200-denoised \
    --max_to_keep=45 \
    --save_frequency=5 \
   train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot3-denoised:extra=noisy_mini-imagenet-shot3-denoised-extra)


### evaluate denoised image (shot3) 14h
for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot3-200-denoised/config.yaml \
        --pretrained-weights output_shot3-200-denoised/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot3-200-denoised/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot3-denoised:extra=noisy_mini-imagenet-shot3-denoised-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot3-denoised:extra=noisy_mini-imagenet-shot3-denoised-extra)
done

# train 130 (shot3)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_shot3_130_config.yaml \
    --output-dir output_shot3-resume-0-130-200-0-70-70 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot3:extra=noisy_mini-imagenet-shot3-extra)


for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot3-resume-0-130-200-0-70-70/config.yaml \
        --pretrained-weights output_shot3-resume-0-130-200-0-70-70/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot3-resume-0-130-200-0-70-70/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot3:extra=noisy_mini-imagenet-shot3-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot3:extra=noisy_mini-imagenet-shot3-extra)
done

# train 140 (shot3)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_shot3_140_config.yaml \
    --output-dir output_shot3-resume-0-140-200-0-60-60 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot3:extra=noisy_mini-imagenet-shot3-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot3-resume-0-140-200-0-60-60/config.yaml \
        --pretrained-weights output_shot3-resume-0-140-200-0-60-60/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot3-resume-0-140-200-0-60-60/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot3:extra=noisy_mini-imagenet-shot3-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot3:extra=noisy_mini-imagenet-shot3-extra)
done


# train 120 (shot3)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_shot3_120_config.yaml \
    --output-dir output_shot3-resume-0-120-200-0-80-80 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot3:extra=noisy_mini-imagenet-shot3-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot3-resume-0-120-200-0-80-80/config.yaml \
        --pretrained-weights output_shot3-resume-0-120-200-0-80-80/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot3-resume-0-120-200-0-80-80/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot3:extra=noisy_mini-imagenet-shot3-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot3:extra=noisy_mini-imagenet-shot3-extra)
done


#####################################################################################

### noisy shot10 22h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_noisy_config.yaml \
    --output-dir output_shot10-200 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot10:extra=noisy_mini-imagenet-shot10-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot10-200/config.yaml \
        --pretrained-weights output_shot10-200/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot10-200/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot10:extra=noisy_mini-imagenet-shot10-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot10:extra=noisy_mini-imagenet-shot10-extra)
done


### denoised image (shot10) 8h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_denoised_config.yaml \
    --output-dir output_shot10-200-denoised \
    --max_to_keep=45 \
    --save_frequency=5 \
   train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot10-denoised:extra=noisy_mini-imagenet-shot10-denoised-extra)


### evaluate denoised image (shot10) 14h
for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot10-200-denoised/config.yaml \
        --pretrained-weights output_shot10-200-denoised/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot10-200-denoised/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot10-denoised:extra=noisy_mini-imagenet-shot10-denoised-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot10-denoised:extra=noisy_mini-imagenet-shot10-denoised-extra)
done

# train 130 (shot10)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_shot10_130_config.yaml \
    --output-dir output_shot10-resume-0-130-200-0-70-70 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot10:extra=noisy_mini-imagenet-shot10-extra)


for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot10-resume-0-130-200-0-70-70/config.yaml \
        --pretrained-weights output_shot10-resume-0-130-200-0-70-70/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot10-resume-0-130-200-0-70-70/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot10:extra=noisy_mini-imagenet-shot10-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot10:extra=noisy_mini-imagenet-shot10-extra)
done

# train 140 (shot10)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_shot10_140_config.yaml \
    --output-dir output_shot10-resume-0-140-200-0-60-60 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot10:extra=noisy_mini-imagenet-shot10-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot10-resume-0-140-200-0-60-60/config.yaml \
        --pretrained-weights output_shot10-resume-0-140-200-0-60-60/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot10-resume-0-140-200-0-60-60/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot10:extra=noisy_mini-imagenet-shot10-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot10:extra=noisy_mini-imagenet-shot10-extra)
done


# train 120 (shot10)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_shot10_120_config.yaml \
    --output-dir output_shot10-resume-0-120-200-0-80-80 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot10:extra=noisy_mini-imagenet-shot10-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot10-resume-0-120-200-0-80-80/config.yaml \
        --pretrained-weights output_shot10-resume-0-120-200-0-80-80/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot10-resume-0-120-200-0-80-80/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot10:extra=noisy_mini-imagenet-shot10-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot10:extra=noisy_mini-imagenet-shot10-extra)
done

##############################################################################################################################


### noisy shot1 22h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_noisy_config.yaml \
    --output-dir output_shot1-200 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot1-200/config.yaml \
        --pretrained-weights output_shot1-200/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot1-200/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)
done


### denoised image (shot1) 8h
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_denoised_config.yaml \
    --output-dir output_shot1-200-denoised \
    --max_to_keep=45 \
    --save_frequency=5 \
   train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1-denoised:extra=noisy_mini-imagenet-shot1-denoised-extra)


### evaluate denoised image (shot1) 14h
for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999 106249 112499 118749 124999 131249 137499 143749 149999 156249 162499 168749 174999 181249 187499 193749 199999 206249 212499 218749 224999 231249 237499 243749 249999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot1-200-denoised/config.yaml \
        --pretrained-weights output_shot1-200-denoised/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot1-200-denoised/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1-denoised:extra=noisy_mini-imagenet-shot1-denoised-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot1-denoised:extra=noisy_mini-imagenet-shot1-denoised-extra)
done

# train 130 (shot1)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_shot1_130_config.yaml \
    --output-dir output_shot1-resume-0-130-200-0-70-70 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)


for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot1-resume-0-130-200-0-70-70/config.yaml \
        --pretrained-weights output_shot1-resume-0-130-200-0-70-70/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot1-resume-0-130-200-0-70-70/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)
done

# train 140 (shot1)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_shot1_140_config.yaml \
    --output-dir output_shot1-resume-0-140-200-0-60-60 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot1-resume-0-140-200-0-60-60/config.yaml \
        --pretrained-weights output_shot1-resume-0-140-200-0-60-60/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot1-resume-0-140-200-0-60-60/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)
done


# train 120 (shot1)
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_shot1_120_config.yaml \
    --output-dir output_shot1-resume-0-120-200-0-80-80 \
    train.dataset_path=ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)

for step in 6249 12499 18749 24999 31249 37499 43749 49999 56249 62499 68749 74999 81249 87499 93749 99999
do
    JOB_ID=$(python dinov2/run/eval/linear.py \
        --config-file output_shot1-resume-0-120-200-0-80-80/config.yaml \
        --pretrained-weights output_shot1-resume-0-120-200-0-80-80/eval/training_$step/teacher_checkpoint.pth \
        --output-dir output_shot1-resume-0-120-200-0-80-80/eval/training_$step/linear \
        --train-dataset ImageNet:split=TRAIN:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra \
        --val-dataset ImageNet:split=VAL:root=noisy_mini-imagenet-shot1:extra=noisy_mini-imagenet-shot1-extra)
done
