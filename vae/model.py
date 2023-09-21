'''
Model file to define architecture of encoder and decoder
'''

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader


# Encoder = takes input from dataset and outputs parameters of normally distributed variational posterior q(z/x)
class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        # Ideally size of sigma should be z_dim*z_dim but since we are making independence assumption we are only giving daigonals of sigma
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.relu = nn.LeakyReLU(0.02)  # figure out good parameter

    def forward(self, x):
        # x = [batch_size, x_dim]
        x = self.fc1(x)
        # x = [batch_size, h_dim]
        x = self.relu(x)
        # x = [batch_size, h_dim]
        mu = self.fc2(x)
        # mu = [batch_size, z_dim]
        sigma = self.fc3(x)
        # sigma = [batch_size, z_dim*z_dim]
        # KL(p//q) = 0.5 * (mu^T*mu + tr{sigma} - z_dim - log(|sigma|))
        self.kl = (self.kl_div_batch(mu, sigma, self.z_dim)).sum()
        # self.kl = [] i.e scalar
        return mu, sigma
    
    def kl_div_batch(self, mu, sigma, z_dim):
        # mu: A batch of mean vectors, with shape (batch_size, latent_dim).
        # sigma: A batch of covariance matrices, with shape (batch_size, latent_dim, latent_dim)
        
        # Calculate the trace of the covariance matrices. 
        # Link : https://discuss.pytorch.org/t/get-the-trace-for-a-batch-of-matrices/108504
        tr_sigma = sigma.sum(dim=1)
        # tr_sigma = [batch_size]

        # Calculate the log determinant of the covariance matrices.
        # I think torch.logdet() takes last 2 dimensions of any tensor and consider them as a matrix's rows and columns
        log_det_sigma = 2 * sigma.log().sum(dim=-1)
        # log_det_sigma = [batch_size]
        
        # Calculate the KL divergence.
        # torch.matmul(mu, mu.t()) = [batch_size, batch_size]
        # mu**2 calculates the square of each element in the tensor mu. mu**2 = [batch_size, z_dim]
        kl_div = 0.5 * ((mu**2).sum(dim=1) + tr_sigma - z_dim - log_det_sigma)
        # kl_div = [batch_size]
        return kl_div

# Decoder = Here the architecture varries according problem in hand. For Q.2 of the assignment we want to reduce 
# MSE for conditional likelihood, so decoder takes samples of z and generate x corresponding to that z
class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim,h_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, z):
        z = self.fc1(z)
        z = self.relu(z)
        z = self.fc2(z)
        # not applying any non linearity here. is it correct?
        return z
    
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.enc = Encoder(x_dim, h_dim, z_dim)
        self.dec = Decoder(x_dim, h_dim, z_dim)

    def forward(self, x):
        mu, sigma = self.enc(x)
        epsilon = torch.rand(sigma.size())
        z = mu + sigma*epsilon
        x = self.dec(z)
        return x
    
def test():
    model = VAE(784, 128, 20)
    # input to nn.Module must be a 'PyTorch tensor'
    x = torch.rand(128, 784)
    x_hat = model(x)
    print(x_hat.shape)
    
if __name__=="__main__":
    test()