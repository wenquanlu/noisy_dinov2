# Python 3.9.16
import os
import glob
import fnmatch
import PIL.Image
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
import random

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument("--random_seed", type=int, default=42)
opt, _ = parser.parse_known_args()

random.seed(opt.random_seed)
#def filter_image_sizes(images):
#    filtered = []
#    for idx, fname in enumerate(images):
#        if (idx % 10000) == 0:
#            print('loading images', idx, '/', len(images))
#        try:
#            with PIL.Image.open(fname) as img:
#                w = img.size[0]
#                h = img.size[1]
#                if (w > 512 or h > 512) or (w < 256 or h < 256):
#                    continue
#                filtered.append(fname)
#        except:
#            print('Could not load image', fname, 'skipping file..')
#    return filtered

def is_valid_image(fname):
    """Check if the image meets the size requirements."""
    try:
        with PIL.Image.open(fname) as img:
            w = img.size[0]
            h = img.size[1]
            return (256 <= w <= 512) and (256 <= h <= 512)
    except Exception as e:
        print(f"Error loading image {fname}: {e}")
        return False

def filter_image_sizes(images):
    """Filter images in parallel."""
    with Pool() as pool:
        results = list(tqdm(pool.imap(is_valid_image, images), total=len(images)))
    return [img for img, valid in zip(images, results) if valid]

def load_and_save(img_path):
    img_name = os.path.basename(img_path)
    img = PIL.Image.open(img_path)
    img = img.convert("RGB")
    save_path = os.path.join(save_dir, img_name)
    img.save(save_path, quality=100, subsampling=0)

def process_images_in_parallel(image_list):
    """Process images using multiprocessing."""
    with Pool() as pool:
        list(tqdm(pool.imap(load_and_save, image_list), total=len(image_list)))

input_dir = opt.input_dir
save_dir = opt.save_dir

images = []
print(input_dir)
pattern = os.path.join(input_dir, '**/*')
all_fnames = glob.glob(pattern, recursive=True)
for fname in all_fnames:
    # include only JPEG/jpg/png
    if fnmatch.fnmatch(fname, '*.JPEG') or fnmatch.fnmatch(
            fname, '*.jpg') or fnmatch.fnmatch(fname, '*.png'):
        images.append(fname)
images = sorted(images)

filtered = filter_image_sizes(images)

print(f"Total filtered images: {len(filtered)}")

# Group images by class
class_images = defaultdict(list)
for img_path in filtered:
    class_name = os.path.basename(img_path).split('_')[0]
    class_images[class_name].append(img_path)

# Randomly select 100 images per class
selected_images = []
for class_name, img_list in class_images.items():
    selected_images.extend(random.sample(img_list, 100))

print(f"Total selected images: {len(selected_images)}")
os.makedirs(save_dir, exist_ok=True)

process_images_in_parallel(selected_images)
#for idx, img_path in tqdm(enumerate(filtered)):
#    if (idx % 1000) == 0:
#        print('loading and saving images', idx, '/', len(filtered))
#    load_and_save(img_path)
#print(len(glob.glob(os.path.join(save_dir, "*.JPEG"))))
