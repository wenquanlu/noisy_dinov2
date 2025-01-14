import torch

print(torch.load("output_clean-200/eval/training_249999/teacher_checkpoint.pth").keys())
print(torch.load("output_clean-200/model_final.rank_0.pth")["model"].keys())
