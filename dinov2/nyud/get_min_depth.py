import os
min_depth = 100000
import numpy as np
for file in os.listdir("/home/wenquan-lu/Workspace/nyuv2-python-toolkit/NYUv2/depth/train"):
    curr_depth = np.min(np.load(f"/home/wenquan-lu/Workspace/nyuv2-python-toolkit/NYUv2/depth/train/{file}"))
    if curr_depth < min_depth:
        min_depth = curr_depth

print(min_depth)

max_depth = -1000000

for file in os.listdir("/home/wenquan-lu/Workspace/nyuv2-python-toolkit/NYUv2/depth/train"):
    curr_depth = np.max(np.load(f"/home/wenquan-lu/Workspace/nyuv2-python-toolkit/NYUv2/depth/train/{file}"))
    if curr_depth > max_depth:
        max_depth = curr_depth

print(max_depth)