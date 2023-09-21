'''
Model file to define architecture of encoder and decoder
'''

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader


# Encoder = takes input from dataset and outputs parameters of normally distributed variational posterior q(z/x)
class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, out_dim)
        self.fc3 = nn.Linear(h_dim, out_dim*out_dim)
        self.relu = nn.LeakyReLU(0.02)

    def forward(self, x):
        # x = [batch_size, in_dim]
        x = self.fc1(x)
        # x = [batch_size, 256]
        x = self.relu(x)
        # x = [batch_size, 256]
        mu = self.fc2(x)
        sigma = self.fc3(x)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return mu, sigma

# Decoder = Here the architecture varries according problem in hand. For Q.2 of the assignment we want to reduce 
# MSE for conditional likelihood, so decoder takes samples of z and generate x corresponding to that z
class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim,256)
        self.fc2 = nn.Linear(256, in_dim)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, z):
        z = self.fc1(z)
        z = self.relu(z)
        z = self.fc2(z)
        return self.relu(z)
    
class VAE(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.enc = Encoder(in_dim, h_dim, out_dim)
        self.dec = Decoder(out_dim, h_dim, in_dim)

    def forward(self, x):
        mu, sigma = self.enc(x)
        epsilon = torch.rand(x.size())
        z = mu + sigma*epsilon
        x = self.dec(x)
        return x
    
def test():
    enc = Encoder(784, 12)
    dec = Decoder(z_dim=12, in_dim=784)
    # input to nn.Module must be a 'PyTorch tensor'
    x = torch.rand(128, 784)
    z_enc = enc(x)
    print(z_enc.shape)
    z = torch.rand(1,12)
    x_dec = dec(z)
    print(x_dec.shape)
    
if __name__=="__main__":
    test()