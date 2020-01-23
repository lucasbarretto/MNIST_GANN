import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=0)

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
        x = F.leaky_relu(self.l1(x), 0.2) # (input, negative_slope=0.2)
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

# function to print sample images
def printImages(images):
    fig = plt.figure(1)
    
    for i in range(len(images)):
        ax = fig.add_subplot(4,5,i+1)
        ax.set_axis_off()
        ax = plt.imshow(images[i].view(28,28).detach().numpy(), cmap='Greys_r')

    plt.show()

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
def getRandomNoise(z_i):
    return torch.rand(1, z_i)

# training hyperparameters
maxEpochs = 100
d_learningRate = 0.001
g_learningRate = 0.001

criterion = nn.BCELoss()  # Binary cross entropy
d_optimizer = optim.SGD(D.parameters(), lr=d_learningRate, momentum=0.8)
g_optimizer = optim.SGD(G.parameters(), lr=g_learningRate, momentum=0.8)

d_loss_data = []
g_loss_data = []
sampleGenImages = []

for epoch in range(maxEpochs):

    for i,data in enumerate(trainloader,0):                
        realImages, realLabels = data
        batchSize = realLabels.size(0)
        d_optimizer.zero_grad()
        
        # train D on real samples (RS = Real Samples)
        d_prediction_RS = D(realImages)
        d_labels_RS = torch.ones([batchSize,1]) # samples belong to the real data distribution
        d_loss_RS = criterion(d_prediction_RS, d_labels_RS)
        d_loss_RS.backward() # compute gradients without changing D's parameters

        # train D on fake samples (FS = Fake Samples)
        z = getRandomNoise(z_i)
        fakeImages = G(z)
        d_prediction_FS = D(fakeImages)
        d_labels_FS = torch.zeros([1,1]) # samples belong to the real data distribution
        d_loss_FS = criterion(d_prediction_FS, d_labels_FS)
        d_loss_FS.backward() # compute gradients without changing D's parameters

        d_loss = d_loss_RS + d_loss_FS
        #d_loss.backward()
        d_optimizer.step()

        # train G
        g_optimizer.zero_grad()
        z = getRandomNoise(z_i)
        fakeImages = G(z)
        d_loss_g = D(fakeImages)
        g_loss = criterion(d_loss_g, torch.ones([1,1]))

        g_loss.backward()
        g_optimizer.step()

    # print losses
    if epoch % 1 == 0:
        print('Epoch: %s - D: (%s) | G: (%s)' % (epoch, d_loss.item(), g_loss.item()))

    if epoch % 5 == 0:
        sampleGenImages.append(fakeImages)
        
    # store losses
    d_loss_data.append(d_loss.item())
    g_loss_data.append(g_loss.item())

# print samples of generated images through training
printImages(sampleGenImages)

# print loss charts
plt.plot(d_loss_data)
plt.plot(g_loss_data)
plt.show()

