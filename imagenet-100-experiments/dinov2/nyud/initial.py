
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from dinov2.hub.depthers import _make_dinov2_dpt_depther
from dinov2.eval.depth.models.losses import GradientLoss, SigLoss

import torch


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])


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


class nyudDataset(Dataset):
    def __init__(self, base_dir, split):
        self.depth_dir = base_dir + "/depth/" + split + "/"
        self.img_dir = base_dir + "/image/" + split + "/"
        self.img_files = os.listdir(self.img_dir)
        self.split = split
        self.transform = make_depth_transform()
        self.flip_transform = transforms.RandomHorizontalFlip(p=1.0)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img = Image.open(self.img_dir + img_file).convert(mode="RGB")
        img = self.transform(img)
        depth = np.load(self.depth_dir + img_file[:-4] + ".npy")
        depth = torch.tensor(depth)
        if self.split == "train" and np.random.rand() > 0.5:
            img = self.flip_transform(img)
            depth = torch.flip(depth, dims=[1])  # Flip horizontally along width dimension
        return img, depth

def eval_model(depth_model, dataloader):
    depth_batches = []
    out_batches = []
    for img, depth in dataloader:
        img = img.cuda()
        depth = depth.cuda()
        depth = depth.unsqueeze(1)
        with torch.inference_mode():
            out = depth_model.encode_decode(img, None)#, rescale=False)
        depth_batches.append(depth)
        out_batches.append(out)
    print("RMSE:", calculate_rmse(depth_batches, out_batches))

NYUd = nyudDataset("/home//Workspace/nyuv2-python-toolkit/NYUv2", "train")
test_set = nyudDataset("/home//Workspace/nyuv2-python-toolkit/NYUv2", "test")

depth_model = _make_dinov2_dpt_depther(arch_name="vit_small", pretrained=False, weights=None, depth_range=(0.7132995, 9.99547))

state_dict = torch.load("output_clean-200/eval/training_249999/teacher_checkpoint.pth")["teacher"]

new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("backbone"):
        new_key = key.replace("backbone.", "", 1)  # Remove the prefix 'teacher.'
        new_state_dict[new_key] = value

depth_model.backbone.load_state_dict(new_state_dict)

dataloader = DataLoader(NYUd, batch_size=16, shuffle=True)
testloader = DataLoader(test_set, batch_size=16, shuffle=False)
num_epochs = 50

optimizer = torch.optim.AdamW(depth_model.parameters(), lr=1e-5, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-4,
    steps_per_epoch=len(dataloader),
    epochs=num_epochs,
    anneal_strategy='cos',
    pct_start=0.1,  # 10% warmup
)

depth_model.cuda()
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    depth_model.train()
    depth_model.backbone.eval()
    for img, depth in dataloader:
        img = img.cuda()
        depth = depth.cuda()
        depth = depth.unsqueeze(1)
        optimizer.zero_grad()
        data_batch = {
            "img": img,
            "img_metas": None,
            "depth_gt": depth
        }
        out = depth_model.train_step(data_batch, optimizer)
        loss = out["loss"]
        #print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
    depth_model.eval()
    eval_model(depth_model, testloader)



torch.save(depth_model.state_dict(), 'dinov2/nyud/testrun1.pth')



