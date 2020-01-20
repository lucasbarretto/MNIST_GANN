import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=0)

import torch.nn as nn
import torch.nn.functional as F

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
        x = F.leaky_relu(self.l1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.l2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.l3(x), 0.2)
        x = self.dropout(x)
        x = self.l4(x)

        return x

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
        x = F.tanh(self.l4(x))

        return x


# discriminator hyperparameters
d_i = 28*28 # discriminator input size
d_n = 32 # discriminator hidden layer size
d_o = 1 # discriminator output size

# generator hyperparameters
z_i = 100 # random distribution input size
g_n = 32 # generator hidden layer size
g_o = 28*28 # generator output size

# initiate the discriminator network
D = Discriminator(d_i, d_n, d_o)

# initiate the generator network
G = Generator(z_i, g_n, g_o)

# generate random noise distribution
def getRandomNoise():
    return torch.rand(1, z_i)
