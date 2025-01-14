import torch


checkpoint = torch.load("output_gauss100-200-denoised/model_0174999.rank_0.pth")
print(checkpoint['model'].keys())