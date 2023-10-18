import os
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

device = torch.device("cuda")
cpu_device = torch.device("cpu")

class Encoder(nn.Module):
    def __init__(self, filters,  kernel_sizes,  strides, hiddens_sizes, paddings, 
                 return_only_conv=False, return_only_liner=False, droput_prob=0.1):
        super(Encoder, self).__init__()
        
        self.return_only_conv = return_only_conv
        self.return_only_liner = return_only_liner
        conv_layers = []
        for i in range(len(kernel_sizes)):
            conv_layers.append(nn.Conv2d(filters[i], filters[i+1], kernel_sizes[i], strides[i], paddings[i]))
            conv_layers.append(nn.ReLU(True))

        self.conv_layer = nn.Sequential(*conv_layers)

        hidden_layers = []
        for i in range(len(hiddens_sizes)-1):
            hidden_layers.append(nn.Dropout(p=droput_prob))
            hidden_layers.append(nn.Linear(hiddens_sizes[i], hiddens_sizes[i+1]))
            hidden_layers.append(nn.ReLU(True))
        self.liner_layer = nn.Sequential(*hidden_layers[:-1])
        
        
    def forward(self, x):
        if self.return_only_conv:
            x = self.conv_layer(x)
            x = x.flatten(start_dim=1)
            return x
        elif self.return_only_liner:
            x = x.flatten(start_dim=1)
            x = self.liner_layer(x)
            return x
        else:
            x = self.conv_layer(x)
            x = x.flatten(start_dim=1)
            x = self.liner_layer(x)
        return x

    
class Decoder(nn.Module):
    def __init__(self, conv_op_size,  filters, kernel_sizes, strides, output_paddings, 
                   paddings, hiddens_sizes, return_only_conv=False, return_only_liner=False, droput_prob=0.1):
        super(Decoder, self).__init__()
        self.return_only_conv = return_only_conv
        self.return_only_liner = return_only_liner
        print(hiddens_sizes)
        
        hidden_layers = []
        for i in range(len(hiddens_sizes)-1):
            hidden_layers.append(nn.Dropout(p=droput_prob))
            hidden_layers.append(nn.Linear(hiddens_sizes[i], hiddens_sizes[i+1]))
            hidden_layers.append(nn.ReLU(True))
        
        self.liner_layer = nn.Sequential(*hidden_layers[:-1])

        conv_layers = []
        for i in range(len(kernel_sizes)):
            conv_layers.append(nn.ConvTranspose2d(filters[i], filters[i+1], kernel_sizes[i], 
                                                  stride=strides[i], output_padding=output_paddings[i], padding=paddings[i]))
            conv_layers.append(nn.ReLU(True))

        self.conv_layer = nn.Sequential(*conv_layers)
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=conv_op_size)
        
        
    def forward(self, x):
        if self.return_only_conv:
            x = self.unflatten(x)
            x = self.conv_layer(x)
            x = torch.sigmoid(x)
            return x
        elif self.return_only_liner:
            # print(x.shape)
            x = self.liner_layer(x)
            x = self.unflatten(x)
            x = torch.sigmoid(x)
            return x
        else:
            x = self.liner_layer(x)
            x = self.unflatten(x)
            x = self.conv_layer(x)
            x = torch.sigmoid(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, feature_size=2048, conv_ip_size=(32, 14, 14), filters = [3,12,24,48,128],  
                 kernel_sizes = [3, 3, 3, 3], strides = [2, 2, 2, 2], output_paddings = [0,0,0,0], 
                 paddings = [0,0,0,0], hiddens_sizes = [2048, 1024, 512, 256, 3], return_only_conv=False, 
                 return_only_liner=False, droput_prob=0.2):
        '''
        if return_only_liner=True, then conv_ip_size = (3, 128, 128) and hidden_sizes [3*128*128, ... , features_size]
        '''
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(filters=filters, 
                               kernel_sizes=kernel_sizes,strides=strides, hiddens_sizes=hiddens_sizes, 
                               return_only_conv=return_only_conv, return_only_liner=return_only_liner, 
                               droput_prob=droput_prob, paddings=paddings)
        
        self.decoder = Decoder(conv_op_size=conv_ip_size, filters=filters[::-1], 
                               kernel_sizes=kernel_sizes[::-1],strides=strides[::-1], output_paddings=output_paddings[::-1],
                                 paddings=paddings[::-1], hiddens_sizes=hiddens_sizes[::-1] , return_only_liner=return_only_liner)
    
    def forward(self,x):
        enc = self.encoder(x)
        x = self.decoder(enc)
        return x, enc

if __name__ == "__main__":
    ae = AutoEncoder().to(device)
    # op, enc = ae(x.to(device))
    # print(op.shape, enc.shape)

    # summary(ae, (3,128,128), device="cuda")

    # e = Encoder()
    # eo = e(x)
    # eo.shape