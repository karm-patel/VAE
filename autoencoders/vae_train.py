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
from torchvision.utils import save_image, make_grid
from torchvision.utils import save_image, make_grid
# Load dataset
device = torch.device("cuda")
cpu_device = torch.device("cpu")


width = 128
train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((width,width))])
# train_transform = transforms.Compose([transforms.ToTensor()])

train_data = AnimalfaceDataset(transform=train_transform, img_width=width, type="train")

val_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((width,width))])
val_data = AnimalfaceDataset(transform=val_transform, type="val", img_width=width)

def show_img(x):
    plt.figure(figsize=(2,2))
    plt.imshow(x.permute(0,2,3,1).detach().to(cpu_device).numpy()[0])

BATCH_SIZE = 512
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
x, y = next(iter(val_loader))
print(x.shape, y.shape)
show_img(x)

# from models.vae import VAE, Encoder, Decoder
from models.vae2 import VAE, Encoder, Decoder


n_samples = 1
feature_size = 32
for beta in [0.01, 0.1, 0.001, 10]:
    n_epochs = 50
    filters = [3, 16, 32, 64, 128, 256]
    kernel_sizes = [4, 4, 4, 4, 4]
    strides = [2,2,2,2,2]
    paddings = [0,0,0,0,0]
    output_paddings = [0,1,0,0,0]
    # paddings = [0,0]
    return_only_liner = 0
    dropout_prob = 0.0


    if return_only_liner:
        conv_ip_size = (3, 128, 128)
        hidden_sizes = [128*128*3, 4096, feature_size]
    else:
        conv_ip_size = (256,2,2)
    #     hidden_sizes = [conv_ip_size[0]*conv_ip_size[1]*conv_ip_size[2], 256*3, 256, feature_size]
        hidden_sizes = [conv_ip_size[0]*conv_ip_size[1]*conv_ip_size[2], feature_size]    

    e = Encoder(filters=filters, 
                kernel_sizes=kernel_sizes,strides=strides, hiddens_sizes=hidden_sizes, 
                return_only_liner=return_only_liner, return_only_conv=0, paddings=paddings, curr_device="cpu")
    # op, mu, logvar = e(x.to(device)).to(device)
    op, mu, logvar = e(x)
    print(op.shape)
    # summary(e, (3,64,64), device="cpu")
    summary(e, (3,width,width), device="cpu")

    vae = VAE(feature_size=feature_size, conv_ip_size=conv_ip_size, filters=filters, 
                    kernel_sizes=kernel_sizes,strides=strides,output_paddings=output_paddings, 
                    paddings=paddings, hiddens_sizes=hidden_sizes, return_only_liner=return_only_liner, 
            droput_prob=dropout_prob, n_samples=n_samples).to(device)
    op, enc, mu, logvar = vae(x.to(device))
    print(op.shape, enc.shape)

    # summary(ae, (3,32,32), device="cuda")
    # summary(vae, (3,width,width), device="cuda")

    
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
            loss, mse_loss, kl_loss = vae.loss_fn(X, X_hat, mu, logvar, beta=beta)
            loss.backward()
            optim.step()
            tqdm_obj.set_description_str(f"Epoch: {epoch} Train Loss {loss}")
            if i%1 == 0:
                losses.append((loss.detach().to(cpu_device), mse_loss.detach().to(cpu_device), kl_loss.detach().to(cpu_device)))
        
        with torch.no_grad(): # mandatory to write
            vae.eval()
            tqdm_obj = tqdm(val_loader)
            for i, (X,y) in enumerate(tqdm_obj):
                X, y = X.to(device), y.to(device)

                X_hat, enc, mu, logvar = vae(X) # [B, feature_size]
                loss, mse_loss, kl_loss = vae.loss_fn(X, X_hat, mu, logvar, beta=beta)

                tqdm_obj.set_description_str(f"Epoch {epoch} Val Loss {loss}")
                if i%1 == 0:
                    val_losses.append((loss.detach().to(cpu_device), mse_loss.detach().to(cpu_device), kl_loss.detach().to(cpu_device)))
                    

    torch.save(vae.state_dict(), f"ckpts/vae_{feature_size}_1l_{beta}_{n_samples}.pt")


    import numpy as np
    to_cpu = lambda arr: [each.detach().to(cpu_device) for each in arr]
    model_name = f"vae_{feature_size}_1l_{beta}_{n_samples}"

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15,5))
    ax1.plot(np.array(val_losses)[:,0][0:], label="Total val loss")
    ax1.legend()

    ax2.plot(np.array(val_losses)[:,1][0:], label="val MSE loss")
    ax2.legend()

    ax3.plot(np.array(val_losses)[:,2][0:], label="val KL loss")
    ax3.legend()
    plt.savefig("figs/losses_" + model_name + ".pdf")


    latents = torch.randn((n_samples, 10*10, feature_size))
    x_hat = vae.decoder(latents.to(device)).mean(dim=0)
    generated_grid = make_grid(x_hat, nrow=10)
    plt.figure(figsize=(10,10))
    plt.imshow(generated_grid.permute(1,2,0).detach().to(cpu_device).numpy())
    save_image(generated_grid,f"images/vae_generated_{feature_size}_1l_{beta}_{n_samples}.png")

    
    # Generation
    grid_size = 10
    # z = torch.rand(grid_size*grid_size, Z_DIM)
    x_hat, enc, _, _ = vae(x.to(device))
    # print(x_hat.shape) # ----> .Size([1, 3, 128, 128])
    # x_hat = x_hat.squeeze() # necessary for printing image
    generated_grid = make_grid(x_hat[:grid_size*grid_size], nrow=grid_size)
    plt.imshow(generated_grid.permute(1,2,0).detach().to(cpu_device).numpy())
    save_image(generated_grid,f"images/vae_reconstrcuted_{feature_size}_1l_{beta}_{n_samples}.png")
    # generated_grid

    generated_grid.permute(1,2,0).shape
