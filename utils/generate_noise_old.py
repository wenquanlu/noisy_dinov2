import os
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm

def gaussian_noise(x, std):
    c = std/255
    x = np.array(x) / 255.
    noisy_image = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    return noisy_image.astype(np.uint8)

def shot_noise(x, lamb):
    x = np.array(x) / 255.
    noisy_image = np.clip(np.random.poisson(x * lamb)/float(lamb), 0, 1) * 255
    return noisy_image.astype(np.uint8)

def speckle_noise(x, c):
    x = np.array(x) / 255.
    noisy_image = np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    return noisy_image.astype(np.uint8)

def add_noise_to_dataset(source_root, target_root, noise):
    np.random.seed(12345)
    # Copy the directory structure
    if os.path.exists(target_root):
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root, ignore=shutil.ignore_patterns('*.*'))
    noise_type, noise_param = noise.split("_")
    noise_param = float(noise_param)
    if noise_type == "gauss":
        add_noise = gaussian_noise
    elif noise_type == "shot":
        add_noise = shot_noise
    elif noise_type == "speckle":
        add_noise = speckle_noise
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
                target_image_path = os.path.join(target_class_path, image_file[:-4] + "png")
                
                # Open image, add noise, and save it
                with Image.open(image_path).convert("RGB") as img:
                    noisy_image = add_noise(img, noise_param)
                    Image.fromarray(noisy_image).save(target_image_path)



add_noise_to_dataset('mini-imagenet', "noisy_mini-imagenet-gauss50", "gauss_50")
add_noise_to_dataset('mini-imagenet', "noisy_mini-imagenet-gauss50", "gauss_100")
add_noise_to_dataset('mini-imagenet', "noisy_mini-imagenet-gauss50", "gauss_255")

add_noise_to_dataset('mini-imagenet', "noisy_mini-imagenet-shot1", "shot_1")
add_noise_to_dataset('mini-imagenet', "noisy_mini-imagenet-shot3", "shot_3")
add_noise_to_dataset('mini-imagenet', "noisy_mini-imagenet-shot10", "shot_10")

add_noise_to_dataset('mini-imagenet', "noisy_mini-imagenet-speckle0.4", "speckle_0.4")
add_noise_to_dataset('mini-imagenet', "noisy_mini-imagenet-speckle0.7", "speckle_0.7")
add_noise_to_dataset('mini-imagenet', "noisy_mini-imagenet-speckle1.0", "speckle_1.0")

