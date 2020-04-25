import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# set device
device = torch.device("cuda")

# generate points from latent space distribution
def generateLatentPoints(z_i, batch_size):
    return torch.randn(batch_size, z_i, device=device)

# function to interpolate two points from the latent space
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
    return z

# function to print sample images
def printImages(images):
    for i in range(len(images)):
        ax = plt.subplot(1,len(images),i+1)
        ax.set_axis_off()
        ax = plt.imshow(images[i].view(28,28).cpu().numpy(), cmap='gray')
    plt.show()

# function to print multiple image lines
def printImageMatrix(imageData, title):
    numRows = len(imageData)
    
    n=0
    fig = plt.figure(1)
    fig.suptitle(title)
    for line in imageData:
        for i in range(len(line)):
            ax = fig.add_subplot(numRows,len(line),n+1)
            ax.set_axis_off()
            ax = plt.imshow(line[i].view(28,28).cpu().numpy(), cmap='gray')
            n+=1 
    plt.show()
        
# function to evaluate model
def evaluateModel(G, z_i, mode='random', interpolation_mode='linear'):
    if mode == 'random':
        z = generateLatentPoints(z_i, 10)
    
    elif mode == 'interpolate':
        [p1,p2] = generateLatentPoints(z_i, 2)
        z = interpolateLatentPoints(G, p1, p2, mode=interpolation_mode)
    
    with torch.no_grad():
        sampleImages = G(z)
    
    return sampleImages

# function to generate labels
def generateLabels(arg, batch_size, smooth='False'):
    
    # label smoothing is on
    if smooth == 'True':
        # true labels are in range [0.8, 1.2]
        if arg == 'True':
            labels = 1.2-0.4*torch.rand(batch_size, 1, device=device)
        
        # false labels are in range [0.0, 0.3]
        if arg == 'False':
            labels = 0.2*torch.rand(batch_size, 1, device=device)
    
    # label smoothing is off
    elif smooth == 'False':
        if arg == 'True':
            # true labels are 1s
            labels = torch.ones(batch_size, 1, device=device)
        if arg == 'False':
            # false labels are 0s
            labels = torch.zeros(batch_size, 1, device=device)
    
    return labels

def printLosses(g_loss, d_loss):
    # print loss charts
    plt.plot(d_loss, label='Discriminator Loss', marker = 'o', c='blue')
    plt.plot(g_loss, label='Generator Loss', marker = 'o', c='orange')
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def train(G, D, g_optimizer, d_optimizer, z_i, trainloader, max_epochs):
    print('started training')
    
    criterion = nn.BCELoss()  # binary cross entropy loss

    # initialize data structures
    d_loss_data = []
    g_loss_data = []
    generatedImages = []
    
    for epoch in range(max_epochs):

        for i,data in enumerate(trainloader,0):                
            realImages, realLabels = data[0].to(device), data[1].to(device)
            batchSize = realLabels.size(0)
            d_optimizer.zero_grad()
            
            # train D on real samples (RS = Real Samples)
            d_prediction_RS = D(realImages)
            d_labels_RS = generateLabels('True', batchSize, smooth='False') # samples belong to the real data
            d_loss_RS = criterion(d_prediction_RS, d_labels_RS) # D loss for real samples
            d_loss_RS.backward() # compute gradients without changing D's parameters

            # train D on fake samples (FS = Fake Samples)
            z = generateLatentPoints(z_i, batchSize)
            fakeImages = G(z)
            d_prediction_FS = D(fakeImages)
            d_labels_FS = generateLabels('False', batchSize, smooth='False') # samples belong to the generated data
            d_loss_FS = criterion(d_prediction_FS, d_labels_FS) # D loss for fake samples
            d_loss_FS.backward() # compute gradients without changing D's parameters
        
            d_loss = d_loss_RS + d_loss_FS
            d_optimizer.step()

            # train G
            g_optimizer.zero_grad()
            z = generateLatentPoints(z_i, batchSize)
            fakeImages = G(z)
            d_loss_g = D(fakeImages)
            d_labels_g = generateLabels('True', batchSize, smooth='False')
            g_loss = criterion(d_loss_g, d_labels_g)

            g_loss.backward()
            g_optimizer.step()

        # print losses
        if epoch % 5 == 0 and epoch != 0:
            printLosses(g_loss_data, d_loss_data)

        if epoch % 1 == 0:
            # print losses for generator and discriminator
            print('Epoch: %s - D: (%s) | G: (%s)' % (epoch, d_loss.item(), g_loss.item()))
            
            # print samples of generated images through training
            sampleImages = evaluateModel(G, z_i, mode='random')
            printImages(sampleImages)
            generatedImages.append(sampleImages)

        # store losses
        d_loss_data.append(d_loss.item())
        g_loss_data.append(g_loss.item())
    
    print('finished training')
    return G, D, g_loss_data, d_loss_data, generatedImages