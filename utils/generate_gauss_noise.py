import os
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm

def gaussian_noise(x):
    c = 1.0
    x = np.array(x) / 255.
    noisy_image = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    return noisy_image.astype(np.uint8)

def add_noise_to_dataset(source_root, target_root):
    np.random.seed(12345)
    # Copy the directory structure
    if os.path.exists(target_root):
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root, ignore=shutil.ignore_patterns('*.*'))

    # Loop through each set (train, test, val)
    for set_name in ['train', 'test', 'val']:
        set_path = os.path.join(source_root, set_name)
        # Loop through each class folder
        for class_folder in tqdm(os.listdir(set_path)):
            class_path = os.path.join(set_path, class_folder)
            target_class_path = os.path.join(target_root, set_name, class_folder)
            # Loop through each image
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                target_image_path = os.path.join(target_class_path, image_file)
                
                # Open image, add noise, and save it
                with Image.open(image_path).convert("RGB") as img:
                    noisy_image = gaussian_noise(img)
                    Image.fromarray(noisy_image).save(target_image_path)

# Paths to the source and target datasets
source_root = 'mini-imagenet'  # Change this path to where your original dataset is stored
target_root = 'noisy_mini-imagenet'  # Path where you want to store the noisy dataset

add_noise_to_dataset(source_root, target_root)
