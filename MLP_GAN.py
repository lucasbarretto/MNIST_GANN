import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0")

transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
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

# generate points from latent space distribution
def generateLatentPoints(z_i, batch_size):
    return torch.randn(batch_size, z_i, device=device)

def interpolateLatentPoints(G, p1, p2, numSamples=10, mode='linear'):
    ratios = np.linspace(0, 1, numSamples)
    
    if mode == 'linear':
        z = []
        for ratio in ratios:
            v = (1.0 - ratio) * p1 + ratio * p2
            z.append(v.tolist())

    elif mode == 'spherical':
        return 0
    
    z = torch.tensor(z).to(device)
    G.eval()
    with torch.no_grad():
        fakeImages = G(z)
    return fakeImages

# function to print sample images
def printImages(images):
    
    for i in range(len(images)):
        ax = plt.subplot(1,len(images),i+1)
        ax.set_axis_off()
        ax = plt.imshow(images[i].view(28,28).cpu().numpy(), cmap='gray')
    plt.show()

# function to generate smooth labels
def generateSmoothLabels(arg, batch_size):
    
    # true labels are in range [0.8, 1.2]
    if arg == 'True':
        labels = 1.2-0.4*torch.rand(batch_size, 1, device=device)
    
    # false labels are in range [0.0, 0.3]
    if arg == 'False':
        labels = 0.2*torch.rand(batch_size, 1, device=device)
    
    return labels
    
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

# training hyperparameters
maxEpochs = 100

criterion = nn.BCELoss()  # Binary cross entropy
d_optimizer = optim.SGD(D.parameters(), lr = 0.001, momentum=0.8)
g_optimizer = optim.SGD(G.parameters(), lr = 0.001, momentum=0.8)

d_loss_data = []
g_loss_data = []

print('started training')
for epoch in range(maxEpochs):

    for i,data in enumerate(trainloader,0):                
        realImages, realLabels = data[0].to(device), data[1].to(device)
        batchSize = realLabels.size(0)
        
        d_optimizer.zero_grad()
        
        # train D on real samples (RS = Real Samples)
        d_prediction_RS = D(realImages)
        d_labels_RS = generateSmoothLabels('True', batchSize) # samples belong to the real data
        d_loss_RS = criterion(d_prediction_RS, d_labels_RS) # D loss for real samples
        d_loss_RS.backward() # compute gradients without changing D's parameters

        # train D on fake samples (FS = Fake Samples)
        z = generateLatentPoints(z_i, batchSize)
        fakeImages = G(z)
        d_prediction_FS = D(fakeImages)
        d_labels_FS = generateSmoothLabels('False', batchSize) # samples belong to the generated data
        d_loss_FS = criterion(d_prediction_FS, d_labels_FS) # D loss for fake samples
        d_loss_FS.backward() # compute gradients without changing D's parameters
       
        d_loss = d_loss_RS + d_loss_FS
        d_optimizer.step()

        # train G
        g_optimizer.zero_grad()
        z = generateLatentPoints(z_i, batchSize)
        fakeImages = G(z)
        d_loss_g = D(fakeImages)
        g_loss = criterion(d_loss_g, torch.ones([batchSize,1]).to(device))

        g_loss.backward()
        g_optimizer.step()

    # print losses
    if epoch % 1 == 0:
        print('Epoch: %s - D: (%s) | G: (%s)' % (epoch, d_loss.item(), g_loss.item()))

    if epoch % 5 == 0:
        # print samples of generated images through training
        z_samples = generateLatentPoints(z_i, 10)
        with torch.no_grad():
            sampleImages = G(z_samples)
        printImages(sampleImages)

    # store losses
    d_loss_data.append(d_loss.item())
    g_loss_data.append(g_loss.item())

# print linear interpolated samples
for i in range(6):
    [p1,p2] = generateLatentPoints(z_i, 2)
    sampleImages = interpolateLatentPoints(G, p1, p2)
    printImages(sampleImages)
    
# print loss charts
plt.plot(d_loss_data, label='Discriminator Loss')
plt.plot(g_loss_data, label='Generator Loss')
plt.title('Training Losses')
plt.legend()
plt.show()

