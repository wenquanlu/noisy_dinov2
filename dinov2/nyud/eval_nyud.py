
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from dinov2.hub.depthers import _make_dinov2_dpt_depther
from dinov2.eval.depth.models.losses import GradientLoss, SigLoss

import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])



class nyudDataset(Dataset):
    def __init__(self, base_dir, split):
        self.depth_dir = base_dir + "/depth/" + split + "/"
        self.img_dir = base_dir + "/image/" + split + "/"
        self.img_files = os.listdir(self.img_dir)
        self.split = split
        self.transform = make_depth_transform()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img = Image.open(self.img_dir + img_file).convert(mode="RGB")
        img = self.transform(img)
        depth = np.load(self.depth_dir + img_file[:-4] + ".npy")
        depth = torch.tensor(depth)
        return img, depth, img_file
    

def calculate_rmse(depth_batches, out_batches):
    total_squared_diff = 0.0
    total_elements = 0

    for depth, out in zip(depth_batches, out_batches):
        squared_diff = (depth - out) ** 2
        total_squared_diff += torch.sum(squared_diff)
        total_elements += torch.numel(squared_diff)
    
    mse = total_squared_diff/total_elements
    rmse = torch.sqrt(mse)
    return rmse


NYUd = nyudDataset("/home/wenquan-lu/Workspace/nyuv2-python-toolkit/NYUv2", "test")

depth_model = _make_dinov2_dpt_depther(arch_name="vit_small", pretrained=False, weights=None, depth_range=(0.7132995, 9.99547))

state_dict = torch.load("dinov2/nyud/testrun3.pth")

depth_model.load_state_dict(state_dict)

dataloader = DataLoader(NYUd, batch_size=16)

depth_model.cuda()

depth_model.eval()
depth_batches = []
out_batches = []
for img, depth, img_file in tqdm(dataloader):
    img = img.cuda()
    depth = depth.cuda()
    depth = depth.unsqueeze(1)
    data_batch = {
        "img": img,
        "img_metas": None,
        "depth_gt": depth
    }
    with torch.inference_mode():
        out = depth_model.encode_decode(img, None)#, rescale=False)
    depth_batches.append(depth)
    out_batches.append(out)

    for i in range(len(img_file)):
        img_name = img_file[i]
        out_depth = out[i].squeeze().cpu().numpy()
        norm = plt.Normalize()
        colored_depth = plt.cm.jet(norm(out_depth))
        plt.imsave(f'dinov2/nyud/predicted_depth_high_res/{img_name}', colored_depth)


print(calculate_rmse(depth_batches, out_batches))









