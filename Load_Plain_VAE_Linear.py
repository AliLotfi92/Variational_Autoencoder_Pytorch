
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import math
import numpy as np
import os
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.filterwarnings('ignore')


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

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_enc=False):
        if no_enc:
            gen_z = Variable(torch.randn(100, z_dim), requires_grad=False)
            gen_z = gen_z.to(device)
            return self.decode(gen_z).view(x.size())

        else:
            stats = self.encode(x.view(-1, 784))
            mu = stats[:, :self.z_dim]
            logvar = stats[:, self.z_dim:]
            z = self.reparametrize(mu, logvar)
            x_recon = self.decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()



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
print('The code is running over', device)


z_dim = 2
gamma = 1

test_set = datasets.MNIST('./tmp/MNIST', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=500, shuffle=True, num_workers=3)

VAE = VAE1().to(device)

VAE.load_state_dict(torch.load('./Saved_Networks/Plain_VAE_Linear'))
print('Network is loaded')

if z_dim == 2:
    batch_size_test = 500
    z_list, label_list = [], []

    for i in range(20):
        x_test, y_test= iter(test_loader).next()
        x_test = Variable(x_test, requires_grad=False).to(device)
        _, _, _, z = VAE(x_test)
        z_list.append(z.cpu().data.numpy())
        label_list.append(y_test.numpy())

    z = np.concatenate(z_list, axis=0)
    label = np.concatenate(label_list)
    sns.kdeplot(z[:, 0], z[:, 1], n_levels=30, cmap='Purples_d')
    plt.show()
    plt.scatter(z[:, 0], z[:, 1], c=label)
    plt.show()
