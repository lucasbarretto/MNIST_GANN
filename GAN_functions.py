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
def generateLatentPoints(batch_size, z_i, net_type):
    
    if net_type == 'MLP':
        z = torch.randn(batch_size, z_i, device=device)
    
    elif net_type == 'CNN':
        z = torch.randn(batch_size, z_i, 1, 1, device=device)
        
    return z

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
    plt.savefig('images/' + 'image:' + str(title) + '.png')
    plt.show()

# function to print model losses
def printLosses(d_loss, g_loss):
    # print loss charts
    plt.plot(d_loss, label='Discriminator Loss', c='blue')
    plt.plot(g_loss, label='Generator Loss', c='orange')
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
        
# function to evaluate model
def evaluateModel(G, z_i, net_type, mode='random', interpolation_mode='linear'):
    if mode == 'random':
        z = generateLatentPoints(10, z_i, net_type)
    
    elif mode == 'interpolate':
        [p1,p2] = generateLatentPoints(2, z_i, net_type)
        z = interpolateLatentPoints(G, p1, p2, mode=interpolation_mode)
    
    with torch.no_grad():
        sampleImages = G(z)
    
    return sampleImages

# function to generate labels
def generateLabels(net_type, arg, batch_size, smooth='False'):
    
    if net_type == 'MLP':
        random = torch.rand(batch_size, 1, device=device)
        ones = torch.ones(batch_size, 1, device=device)
        zeros = torch.zeros(batch_size, 1, device=device)
    
    elif net_type == 'CNN':
        random = torch.rand(batch_size, 1, 1, 1, device=device)
        ones = torch.ones(batch_size, 1, 1, 1, device=device)
        zeros = torch.zeros(batch_size, 1, 1, 1, device=device)
        
    # label smoothing is on
    if smooth == 'True':
        # true labels are in range [0.8, 1.2]
        if arg == 'True':
            labels = 1.2-0.4*random
        
        # false labels are in range [0.0, 0.3]
        if arg == 'False':
            labels = 0.2*random
    
    # label smoothing is off
    elif smooth == 'False':
        if arg == 'True':
            # true labels are 1s
            labels = ones
        if arg == 'False':
            # false labels are 0s
            labels = zeros
    
    return labels
    
def train(net_type, trainloader, D, G, d_optimizer, g_optimizer, z_i, max_epochs):
    print('started training')
    
    criterion = nn.BCELoss()  # binary cross entropy loss

    # initialize data structures
    d_loss_data = []
    d_loss_avg = []
    g_loss_data = []
    g_loss_avg = []
    generatedImages = []
    
    for epoch in range(max_epochs):

        for i,data in enumerate(trainloader,0):                
            realImages, realLabels = data[0].to(device), data[1].to(device)
            batchSize = realLabels.size(0)
            d_optimizer.zero_grad()
            
            # train D on real samples (RS = Real Samples)
            d_prediction_RS = D(realImages)
            d_labels_RS = generateLabels(net_type, 'True', batchSize, smooth='False') # samples belong to the real data
            d_loss_RS = criterion(d_prediction_RS, d_labels_RS) # D loss for real samples
            d_loss_RS.backward() # compute gradients without changing D's parameters
            
            # train D on fake samples (FS = Fake Samples)
            z = generateLatentPoints(batchSize, z_i, net_type)
            fakeImages = G(z)
            d_prediction_FS = D(fakeImages)
            d_labels_FS = generateLabels(net_type, 'False', batchSize, smooth='False') # samples belong to the generated data
            d_loss_FS = criterion(d_prediction_FS, d_labels_FS) # D loss for fake samples
            d_loss_FS.backward() # compute gradients without changing D's parameters
        
            d_loss = d_loss_RS + d_loss_FS
            d_optimizer.step()

            # train G
            g_optimizer.zero_grad()
            z = generateLatentPoints(batchSize, z_i, net_type)
            fakeImages = G(z)
            d_loss_g = D(fakeImages)
            d_labels_g = generateLabels(net_type, 'True', batchSize, smooth='False')
            g_loss = criterion(d_loss_g, d_labels_g)

            g_loss.backward()
            g_optimizer.step()
            
            # store losses for each iteration
            d_loss_data.append(d_loss.item())
            g_loss_data.append(g_loss.item())
            
            print(i)
            print('Epoch: %s - D: (%s) | G: (%s)' % (epoch, d_loss.item(), g_loss.item()))
        # print losses
        if epoch % 1 == 0 and epoch != 0:
            printLosses(d_loss_data, g_loss_data)

        if epoch % 1 == 0:
            # print losses for generator and discriminator
            print('Epoch: %s - D: (%s) | G: (%s)' % (epoch, d_loss.item(), g_loss.item()))
            
            # print samples of generated images through training
            sampleImages = evaluateModel(G, z_i, net_type, mode='random')
            printImages(sampleImages)
            generatedImages.append(sampleImages)

        # store avg epoch loss
        d_loss_avg.append(np.mean(d_loss_data))
        g_loss_avg.append(np.mean(g_loss_data))
    
    print('finished training')
    return D, G, d_loss_avg, g_loss_avg, generatedImages