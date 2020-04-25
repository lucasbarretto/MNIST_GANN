import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import GAN_functions as gan

# set device
device = torch.device("cuda")

transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=0)

print('loaded modules and dataset')

# define discriminator class
class Discriminator(nn.Module):
    def __init__(self,i,n,o):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(i,4*n,bias=True)
        self.l2 = nn.Linear(4*n,2*n, bias=True)
        self.l3 = nn.Linear(2*n,n, bias=True)
        self.l4 = nn.Linear(n,o, bias=True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.leaky_relu(self.l1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.l2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.l3(x), 0.2)
        x = self.dropout(x)
        x = torch.sigmoid(self.l4(x))
        return x

# define generator class
class Generator(nn.Module):
    def __init__(self,i,n,o):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(i,n,bias=True)
        self.l2 = nn.Linear(n,2*n, bias=True)
        self.l3 = nn.Linear(2*n,4*n, bias=True)
        self.l4 = nn.Linear(4*n,o, bias=True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.l2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.l3(x), 0.2)
        x = self.dropout(x)
        x = torch.tanh(self.l4(x))
        return x

# discriminator hyperparameters
d_i = 28*28 # discriminator input size
d_n = 21 # discriminator hidden layer size
d_o = 1 # discriminator output size

# generator hyperparameters
z_i = 100 # random distribution input size
g_n = 32 # generator hidden layer size
g_o = 28*28 # generator output size

# initiate the discriminator network
D = Discriminator(d_i, d_n, d_o).to(device)

# initiate the generator network
G = Generator(z_i, g_n, g_o).to(device)

# define optimizers
d_optimizer = optim.SGD(D.parameters(), lr = 0.001, momentum=0.8)
g_optimizer = optim.SGD(G.parameters(), lr = 0.001, momentum=0.8)

# training hyperparameters
maxEpochs = 5

G, D, g_loss, d_loss, generatedImages = gan.train(G, D, g_optimizer, d_optimizer,
                                              z_i, trainloader, maxEpochs)

gan.printLosses(g_loss, d_loss)
gan.printImageMatrix(generatedImages, 'Learning Evolution - Random Samples')

# print interpolated samples
interpolatedSamples = []
for _ in range(4):
    interpolatedSamples.append(gan.evaluateModel(G, z_i, interpolation_mode='interpolate'))
gan.printImageMatrix(interpolatedSamples, 'interpolated samples')

