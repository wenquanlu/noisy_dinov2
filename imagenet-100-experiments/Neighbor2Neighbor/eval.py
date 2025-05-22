from tqdm import tqdm
import argparse
from arch_unet import UNet
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
import os
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument('--state_dict', type=str)
parser.add_argument('--data_dir', type=str, default='./noisy_mini-imagenet-gauss100')
parser.add_argument('--save_dir', type=str, default='./noisy_mini-imagenet-gauss100-denoised')

opt, _ = parser.parse_known_args()

def replace_root_os_path(original_path, new_root):
    # Split the original path
    path_parts = original_path.split(os.sep)
    # Replace the root with the new one
    new_path = os.path.join(new_root, *path_parts[2:])
    return new_path

class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_dir):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.train_fns = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                # Get full path of the file
                if file.endswith("png"):
                    file_path = os.path.join(root, file)
                    self.train_fns.append(file_path)
        self.train_fns.sort()
        #self.train_fns = self.train_fns[51277:]
        print('fetch {} samples for testing'.format(len(self.train_fns)))

    def __len__(self):
        return len(self.train_fns)
    
    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = Image.open(fn).convert("RGB")
        im = np.array(im, dtype=np.float32)/255.0

        return im, fn

    
inferenceDataset = DataLoader_Imagenet_val(opt.data_dir)
loader = DataLoader(dataset=inferenceDataset,
                            num_workers=8,
                            batch_size=1,
                            shuffle=False)

network = UNet(in_nc=opt.n_channel,
               out_nc=opt.n_channel,
               n_feature=opt.n_feature)

network.load_state_dict(torch.load(opt.state_dict))
network.cuda()
network.eval()
try:
    for im, im_path in tqdm(loader):
        im = im.squeeze()
        H = im.shape[0]
        W = im.shape[1]
        if H > 2048 and W > 2048:
            l_size = (max(H, W) + 1023)//1024 * 1024
            im = np.pad(
                im,
                [[0, l_size - H], [0, l_size - W], [0, 0]],
                'reflect')
            count = int(l_size/1024)
            im_combined = np.zeros((1, 3, l_size, l_size))
            transformer = transforms.Compose([transforms.ToTensor()])
            for i in range(count):
                for j in range(count):
                    patch = transformer(im[i*1024:(i+1) * 1024, j*1024:(j+1)*1024, :])
                    patch = torch.unsqueeze(patch, 0)
                    patch=patch.cuda()
                    with torch.no_grad():
                        prediction_patch = network(patch).cpu().data
                    im_combined[:,:, i*1024:(i+1) * 1024, j*1024:(j+1)*1024] = prediction_patch
            prediction = im_combined[:,:,:H,:W].transpose(0, 2, 3, 1).clip(0, 1).squeeze()
            pred255 = np.clip(prediction * 255.0 + 0.5, 0,
                                            255).astype(np.uint8)
            pred255 = Image.fromarray(pred255)
            # we need to initialize the directory structure first
            pred255.save(replace_root_os_path(im_path[0], opt.save_dir))
                    
        else:
            val_size = (max(H, W) + 31) // 32 * 32
            im = np.pad(
                im,
                [[0, val_size - H], [0, val_size - W], [0, 0]],
                'reflect')
            transformer = transforms.Compose([transforms.ToTensor()])
            im = transformer(im)
            im = torch.unsqueeze(im, 0)
            im=im.cuda()
            with torch.no_grad():
                prediction = network(im)
                prediction = prediction[:, :, :H, :W]
            prediction = prediction.permute(0, 2, 3, 1)
            prediction = prediction.cpu().data.clamp(0, 1).numpy()
            prediction = prediction.squeeze()
            pred255 = np.clip(prediction * 255.0 + 0.5, 0,
                                            255).astype(np.uint8)
            pred255 = Image.fromarray(pred255)
            # we need to initialize the directory structure first
            pred255.save(replace_root_os_path(im_path[0], opt.save_dir))
except Exception as e:
    print(im_path)
    print(e)
    exit()
