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
import numpy
import pickle
from transforms import make_classification_eval_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument('--state_dict', type=str)
parser.add_argument('--data_root', type=str, default="/home/wenquan-lu/Workspace/revisitop/data")
parser.add_argument('--noise', type=str)
parser.add_argument('--test_dataset', type=str)

opt, _ = parser.parse_known_args()

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

def identity(x, c, rng):
    return x

class OxfordParisDataset(Dataset):
    def __init__(self, gnd_path, data_root, transform, noise, query=False, seed=42):
        self.gnd_path = gnd_path
        self.data_root = data_root
        self.query = query
        with open(gnd_path, "rb") as f:
            gnd = pickle.load(f)
            if self.query:
                self.data = gnd["qimlist"]
            else:
                self.data = gnd["imlist"]
        self.transform = transform
        self.rng = numpy.random.default_rng(seed)
        noise_type, noise_param = noise.split("_")
        self.noise_param = float(noise_param)
        if noise_type == "gauss":
            self.noise = gaussian_noise
        elif noise_type == "shot":
            self.noise = shot_noise
        elif noise_type == "speckle":
            self.noise = speckle_noise
        elif noise_type == "identity":
            self.noise = identity

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):
        file_name = self.data_root + "/" + self.data[index] + ".jpg"
        img = Image.open(file_name).convert(mode="RGB")
        img = self.noise(img, self.noise_param, self.rng)
        #img = self.transform(img)
        
        return img, self.data[index]

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels

img_data_root = os.path.join(opt.data_root, "datasets", opt.test_dataset, "jpg")
transform = make_classification_eval_transform()

dataset_pkl = os.path.join(opt.data_root, "datasets", opt.test_dataset, "gnd_"+opt.test_dataset+".pkl")
database_dataset = OxfordParisDataset(dataset_pkl, img_data_root, transform, opt.noise, query=False, seed=42)

noise = "".join(opt.noise.split("_"))

database_data_loader = torch.utils.data.DataLoader(
    database_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    persistent_workers=False,
    collate_fn=custom_collate_fn,
)

query_dataset = OxfordParisDataset(dataset_pkl, img_data_root, transform, opt.noise, query=True, seed=100)

query_data_loader = torch.utils.data.DataLoader(
    query_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    persistent_workers=False,
    collate_fn=custom_collate_fn,
)

network = UNet(in_nc=opt.n_channel,
               out_nc=opt.n_channel,
               n_feature=opt.n_feature)

network.load_state_dict(torch.load(opt.state_dict))
network.cuda()
network.eval()
try:
    for ims, im_paths in tqdm(database_data_loader):
        for batch_num in range(len(ims)):
            im = ims[batch_num]
            im_path = im_paths[batch_num]
            #im = im.squeeze()
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
                pred255.save(opt.test_dataset + "_" + noise + "/jpg/" + im_path + ".png")
                        
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
                pred255.save(opt.test_dataset + "_" + noise + "/jpg/" + im_path + ".png")
except Exception as e:
    #print(im_paths)
    print(e)
    exit()


try:
    for ims, im_paths in tqdm(query_data_loader):
        for batch_num in range(len(ims)):
            im = ims[batch_num]
            im_path = im_paths[batch_num]
        #im = im.squeeze()
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
                pred255.save(opt.test_dataset + "_" + noise + "/jpg/" + im_path + ".png")
                        
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
                pred255.save(opt.test_dataset + "_" + noise + "/jpg/" + im_path + ".png")
except Exception as e:
    #print(im_paths)
    print(e)
    exit()
