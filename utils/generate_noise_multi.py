import os
import hashlib
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def gaussian_noise(x, std, rng):
    c = std/255
    x = np.array(x) / 255.
    noisy_image = np.clip(x + rng.normal(size=x.shape, scale=c), 0, 1) * 255
    return noisy_image.astype(np.uint8)

def shot_noise(x, lamb, rng):
    x = np.array(x) / 255.
    noisy_image = np.clip(rng.poisson(x * lamb)/float(lamb), 0, 1) * 255
    return noisy_image.astype(np.uint8)

def speckle_noise(x, c, rng):
    x = np.array(x) / 255.
    noisy_image = np.clip(x + x * rng.normal(size=x.shape, scale=c), 0, 1) * 255
    return noisy_image.astype(np.uint8)

def generate_seed(image_path, base_seed, root_path):
    """Generates a deterministic, platform-independent seed."""
    relative_path = os.path.relpath(image_path, root_path)  # Relative path to root
    hasher = hashlib.md5()
    hasher.update(f"{relative_path}_{base_seed}".encode('utf-8'))
    return int(hasher.hexdigest(), 16) % (2**32)


def process_image(image_info):
    image_path, target_image_path, noise_func, noise_param, seed = image_info
    try:
        rng = np.random.default_rng(seed)  # Create a unique RNG for this task
        with Image.open(image_path).convert("RGB") as img:
            noisy_image = noise_func(img, noise_param, rng)
            Image.fromarray(noisy_image).save(target_image_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def add_noise_to_dataset(source_root, target_root, noise):
    base_seed = 42  # Base seed for reproducibility

    if os.path.exists(target_root):
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root, ignore=shutil.ignore_patterns("*.*"))

    noise_type, noise_param = noise.split("_")
    noise_param = float(noise_param)
    noise_func = {
        "gauss": gaussian_noise,
        "shot": shot_noise,
        "speckle": speckle_noise
    }[noise_type]

    tasks = []
    for set_name in ["train", "val"]:
        set_path = os.path.join(source_root, set_name)
        target_set_path = os.path.join(target_root, set_name)
        for class_folder in tqdm(os.listdir(set_path), desc=f"Processing {set_name}"):
            class_path = os.path.join(set_path, class_folder)
            target_class_path = os.path.join(target_set_path, class_folder)
            os.makedirs(target_class_path, exist_ok=True)

            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                target_image_path = os.path.join(target_class_path, os.path.splitext(image_file)[0] + ".png")
                seed = generate_seed(image_path, base_seed, source_root)  # Use relative path for seed
                tasks.append((image_path, target_image_path, noise_func, noise_param, seed))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, tasks), total=len(tasks), desc="Adding noise"))


# Example usage
add_noise_to_dataset("imagenet", "imagenet-gauss100", "gauss_100")