import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import zipfile
# from natsort import natsorted
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
# from model import VAE

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchmetrics
from dataloader.animal_faces import AnimalfaceDataset
from models.vae2 import VAE, Encoder, Decoder


# Load dataset
device = torch.device("cuda")
cpu_device = torch.device("cpu")

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

width = 128
train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((width,width))])
# train_transform = transforms.Compose([transforms.ToTensor()])

train_data = AnimalfaceDataset(transform=train_transform, img_width=width, type="val")

# val_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((width,width))])
# val_data = AnimalfaceDataset(transform=val_transform, type="val", img_width=width)

def show_img(x):
    plt.figure(figsize=(2,2))
    plt.imshow(x.permute(0,2,3,1).detach().to(cpu_device).numpy()[0])

BATCH_SIZE = 512
# val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
x, y = next(iter(train_loader))
print(x.shape, y.shape)
# show_img(x)

# %reload_ext autoreload
# %autoreload 2

# conv_ip_size = (128, 6, 6)
# feature_size = 2048 
# filters = [3,12,24,48,128]
# kernel_sizes = [7, 5, 3, 3]
# strides = [2, 2, 2, 2]
# output_paddings = [1,0,0,1]
# paddings = [0,0,0,0]


# feature_size = 128 
# filters = [3, 16, 32]
# kernel_sizes = [7,5]
# strides = [2,2]
# output_paddings = [0,1]
# paddings = [0,0]
# return_only_liner = 0
# dropout_prob = 0.05

# feature_size = 20 
# filters = [3, 32, 32, 64, 64, 256]
# kernel_sizes = [4, 4, 4, 4, 4]
# strides = [2,2,2,2,1]
# paddings = [1,1,1,1,0]
# output_paddings = [0,0,0,0,0]
# # paddings = [0,0]
# return_only_liner = 0
# dropout_prob = 0.05



feature_size = 32
filters = [3, 16, 32, 64, 128, 256]
kernel_sizes = [4, 4, 4, 4, 4]
strides = [2,2,2,2,2]
paddings = [0,0,0,0,0]
output_paddings = [0,1,0,0,0]
# paddings = [0,0]
return_only_liner = 0
dropout_prob = 0.0


# Puspak - 64
# feature_size = 32
# filters = [3, 32, 64, 128, 256]
# kernel_sizes = [4, 4, 4, 4]
# strides = [2,2,2,2]
# paddings = [0,0,0,0]
# output_paddings = [0,1,0,0]
# # paddings = [0,0]
# return_only_liner = 0
# dropout_prob = 0.0

if return_only_liner:
    conv_ip_size = (3, 128, 128)
    hidden_sizes = [128*128*3, 4096, feature_size]
else:
    conv_ip_size = (256,2,2)
#     hidden_sizes = [conv_ip_size[0]*conv_ip_size[1]*conv_ip_size[2], 256*3, 256, feature_size]
    hidden_sizes = [conv_ip_size[0]*conv_ip_size[1]*conv_ip_size[2], feature_size]    

# e = Encoder(filters=filters, 
#             kernel_sizes=kernel_sizes,strides=strides, hiddens_sizes=hidden_sizes, 
#             return_only_liner=return_only_liner, return_only_conv=0, paddings=paddings, curr_device="cpu")
# # op, mu, logvar = e(x.to(device)).to(device)
# op, mu, logvar = e(x)
# print(op.shape)
# # summary(e, (3,64,64), device="cpu")
# summary(e, (3,width,width), device="cpu")

# d = Decoder(conv_op_size=conv_ip_size, filters=filters[::-1], 
#                                kernel_sizes=kernel_sizes[::-1],strides=strides[::-1], output_paddings=output_paddings[::-1],
#                                  paddings=paddings[::-1], hiddens_sizes=hidden_sizes[::-1] , return_only_conv=True)
# do = d(op.flatten(start_dim=1))
# print("hello-",do.shape)
# # summary(d, (feature_size,), device="cpu")
# summary(d, op.flatten(start_dim=1).shape, device="cpu")

vae = VAE(feature_size=feature_size, conv_ip_size=conv_ip_size, filters=filters, 
                 kernel_sizes=kernel_sizes,strides=strides,output_paddings=output_paddings, 
                 paddings=paddings, hiddens_sizes=hidden_sizes, return_only_liner=return_only_liner, droput_prob=dropout_prob).to(device)
op, enc, mu, logvar = vae(x.to(device))
print(op.shape, enc.shape)

# summary(ae, (3,32,32), device="cuda")
summary(vae, (3,width,width), device="cuda")


n_epochs = 100

optim = torch.optim.Adam(vae.parameters(), lr=1e-3)
losses = []
val_losses = []
for epoch in range(n_epochs):
    vae.train()
    tqdm_obj = tqdm(train_loader)
    for i, (X,y) in enumerate(tqdm_obj):
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        
        X_hat, enc, mu, logvar = vae(X) # [B, feature_size]
        loss = vae.loss_fn(X, X_hat, mu, logvar, beta=1)
        loss.backward()
        optim.step()

        tqdm_obj.set_description_str(f"Epoch: {epoch} Loss {loss}")
        if i%1 == 0:
            losses.append(loss.detach().to(cpu_device))
    
    with torch.no_grad(): # mandatory to write
        vae.eval()
        tqdm_obj = tqdm(train_loader)
        for i, (X,y) in enumerate(tqdm_obj):
            X, y = X.to(device), y.to(device)

            X_hat, enc, mu, logvar = vae(X) # [B, feature_size]
            val_loss = vae.loss_fn(X, X_hat, mu, logvar, beta=1)

            tqdm_obj.set_description_str(f"Epoch {epoch} Val Loss {val_loss}")
            if i%1 == 0:
                val_losses.append(val_loss.detach().to(cpu_device))
                