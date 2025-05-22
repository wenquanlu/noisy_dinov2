import shutil
import sys

noise = sys.argv[1]

shutil.copytree(f"noisy_mini-imagenet-{noise}", f"noisy_mini-imagenet-{noise}-denoised", ignore=shutil.ignore_patterns('*.*'))
