

import os
import torch
import random
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler



class Discriminator(nn.Module):
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1),
            nn.Sigmoid()

        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()


class FactorVAE1(nn.Module):

    def __init__(self, z_dim=10):
        super(FactorVAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(1, 28, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(28, 28, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(28, 56, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(56, 118, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(118, 2 * z_dim, 1),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 118, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(118, 118, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(118, 56, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(56, 28, 4, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(28, 28, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(28, 1, 4, 2, 1),
            nn.Sigmoid(),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):

        initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):

        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def recon_loss(x_recon, x):
    n = x.size(0)
    loss = F.binary_cross_entropy(x_recon, x, size_average=False).div(n)
    return loss


def kl_divergence(mu, logvar):
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
    return kld

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
max_iter = int(3000)
batch_size = 60
z_dim = 10
lr_D = 0.001
beta1_D = 0.9
beta2_D = 0.999
gamma = 6.4

training_set = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

VAE = FactorVAE1().to(device)
D = Discriminator().to(device)
optim_VAE = optim.Adam(VAE.parameters(), lr=lr_D, betas=(beta1_D, beta2_D))
optim_D = optim.Adam(D.parameters(), lr=lr_D, betas=(beta1_D, beta2_D))


ones = torch.ones(batch_size, dtype=torch.float, device=device)
zeros = torch.zeros(batch_size, dtype=torch.float, device=device)


for epoch in range(max_iter):
    train_loss = 0

    for batch_idx, (x_true1,_) in enumerate(data_loader):

        x_true1 = x_true1.to(device)

        x_recon, mu, logvar, z = VAE(x_true1)

        vae_recon_loss = recon_loss(x_recon, x_true1)
        vae_kld = kl_divergence(mu, logvar)

        D_z = D(z)
        vae_tc_loss = (1 - D_z).mean()
        vae_loss = vae_recon_loss + vae_kld + gamma * vae_tc_loss
        train_loss += vae_loss.item()

        optim_VAE.zero_grad()
        vae_loss.backward(retain_graph=True)
        optim_VAE.step()

        z_prime = Variable(torch.randn(batch_size, z_dim), requires_grad=False).to(device)
        D_z_pperm = D(z_prime)
        D_tc_loss = 0.5 * (F.binary_cross_entropy(D_z_pperm, ones) + F.binary_cross_entropy(D_z, zeros))

        optim_D.zero_grad()
        D_tc_loss.backward()
        optim_D.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t  Loss: {:.6f} '.format(epoch, batch_idx * len(x_true1), len(data_loader.dataset),100. * batch_idx / len(data_loader),vae_loss.item() / len(x_true1)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
    
    
    
    
    
    
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
    
    
    
    
    
