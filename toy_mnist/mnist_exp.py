import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import argparse

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--weight", default="")
    parser.add_argument("--noisy", action="store_true", help="Enable noisy mode")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--save", default="")
    return parser


# Linear model
class LinearHead(nn.Module):
    def __init__(self, input_dim=28*28, proj_dim=32):
        super().__init__()
        self.linear = nn.Linear(input_dim, proj_dim, bias=False)
    def forward(self, x):
        return self.linear(x)



class MLPHead(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim, bias=False)
        )

    def forward(self, x):
        return self.net(x)
    
# Data augmentation with Gaussian noise
def gaussian_noise(img, std=0.4):
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, 0., 1.)

aug = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=10),
])

class AugmentedMNIST(torch.utils.data.Dataset):
    def __init__(self, noisy, train=True):
        self.dataset = datasets.MNIST(root='./data', train=train, download=True)
        self.noisy = noisy
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        img = self.to_tensor(img) 
        if self.noisy:
            img1 = aug(gaussian_noise(img))
            img2 = aug(gaussian_noise(img))
        else:
            img1 = aug(img)
            img2 = aug(img)
        return img1.view(-1), img2.view(-1)
    
# Losses
def invariance_loss(z1, z2):
    return ((z1 - z2)**2).sum(dim=1).mean()

def covariance_penalty(z):
    n, d = z.size()
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / n
    I = torch.eye(d).to(z.device)
    return ((cov - I)**2).sum()

if __name__ == "__main__":

    args = get_args_parser().parse_args()
        

    # Training setup
    pretrain_loader = DataLoader(AugmentedMNIST(args.noisy), batch_size=256, shuffle=True)

    model = MLPHead().cuda()
    if args.weight != "":
        model.load_state_dict(torch.load(args.weight))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lambda_cov = 0.5

    # Training loop
    for epoch in range(args.epoch):
        running_inv_loss = 0.0
        running_cov_loss = 0.0
        for i, (x1, x2) in enumerate(pretrain_loader):
            x1, x2 = x1.cuda(), x2.cuda()
            z1, z2 = model(x1), model(x2)

            inv_loss = invariance_loss(z1, z2)
            cov_loss = covariance_penalty(torch.cat([z1, z2], dim=0))
            loss = inv_loss + lambda_cov * cov_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_inv_loss += inv_loss.item()
            running_cov_loss += cov_loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epoch}], Step [{i+1}/{len(pretrain_loader)}], "
                    f"Invariance Loss: {inv_loss.item():.4f}, Covariance Penalty: {cov_loss.item():.4f}")
        if (epoch + 1)%10 == 0:
            torch.save(model.state_dict(), args.save + f"_{epoch + 1}.pth")

        print(f"Epoch {epoch+1}: Avg Invariance Loss = {running_inv_loss / len(pretrain_loader):.4f}, "
            f"Avg Cov Penalty = {running_cov_loss / len(pretrain_loader):.4f}")
    torch.save(model.state_dict(), args.save + f"_final.pth")

