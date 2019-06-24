
import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler


class Discriminator(nn.Module):
    def __init__(self, z_dim=2):
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
            nn.Linear(1000, 2),
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


class VAE1(nn.Module):

    def __init__(self, z_dim=2):
        super(VAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Linear(784, 400),
            nn.LeakyReLU(0.2, True),
            nn.Linear(400, 400),
            nn.LeakyReLU(0.2, True),
            nn.Linear(400, 2 * z_dim),
        )
        self.decode = nn.Sequential(
            nn.Linear(z_dim, 400),
            nn.LeakyReLU(0.2, True),
            nn.Linear(400, 400),
            nn.LeakyReLU(0.2, True),
            nn.Linear(400, 784),
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

    def forward(self, x, no_enc=False, no_dec=False):

        stats = self.encode(x.view(-1, 784))
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_enc:
            z = Variable(torch.randn(100, z_dim), requires_grad=False).to(device)
            return self.decode(z).view(x.size())

        elif no_dec:
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

def convert_to_display(samples):
    cnt, height, width = int(math.floor(math.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height*cnt, width*cnt])
    return samples


use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
max_iter = int(3000)
batch_size = 100
z_dim = 2
lr_D = 0.001
beta1_D = 0.9
beta2_D = 0.999
gamma = 6.4

training_set = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())

data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=500, shuffle=True, num_workers=3)

VAE = VAE1().to(device)
D = Discriminator().to(device)
optim_VAE = optim.Adam(VAE.parameters(), lr=lr_D, betas=(beta1_D, beta2_D))
optim_D = optim.Adam(D.parameters(), lr=lr_D, betas=(beta1_D, beta2_D))

ones = torch.ones(batch_size, dtype=torch.long, device=device)
zeros = torch.zeros(batch_size, dtype=torch.long, device=device)

for epoch in range(max_iter):
    train_loss = 0

    for batch_idx, (x_true, _) in enumerate(data_loader):

        x_true1 = x_true.to(device)
        x_recon, mu, logvar, z = VAE(x_true)

        vae_recon_loss = recon_loss(x_recon, x_true)
        vae_kld = kl_divergence(mu, logvar)

        D_z = D(z)

        vae_tc_loss = -(D_z[:, :1] - D_z[:, 1:]).mean()
        vae_loss = vae_recon_loss + gamma * vae_tc_loss
        train_loss += vae_loss.item()

        optim_VAE.zero_grad()
        vae_loss.backward(retain_graph=True)
        optim_VAE.step()

        z_prime = Variable(torch.randn(batch_size, z_dim), requires_grad=False).to(device)
        D_z_pperm = D(z_prime)

        D_loss = 0.5 * (F.cross_entropy(D_z_pperm, zeros) + F.cross_entropy(D_z, ones))
        optim_D.zero_grad()
        D_loss.backward()
        optim_D.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t  Loss: {:.6f} \t Discriminator Loss: {:.6f} \t Generator Loss: {:.6f}'.format(epoch, batch_idx * len(x_true1),
                                                                               len(data_loader.dataset),
                                                                               100. * batch_idx / len(data_loader),
                                                                               vae_loss.item(),
                                                                               D_loss.item(),
                                                                               vae_tc_loss.item()
                                                                                               ))
        if batch_idx % 1000 == 0:

            samples = VAE(x_true, no_enc=True)
            samples = samples.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()
            plt.imshow(convert_to_display(samples), cmap='Greys_r')
            plt.show()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))

    if z_dim == 2:
        batch_size_test = 500
        z_list, label_list = [], []

        for i in range(20):
            x_test, y_test = iter(test_loader).next()
            x_test = Variable(x_test, requires_grad=False).to(device)
            z = VAE(x_test, no_dec=True)
            z_list.append(z.cpu().data.numpy())
            label_list.append(y_test.numpy())

        z = np.concatenate(z_list, axis=0)
        label = np.concatenate(label_list)
        plt.scatter(z[:, 0], z[:, 1], c=label)
        plt.show()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
