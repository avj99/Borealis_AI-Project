import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import logging
logging.basicConfig(filename='Large_Google_001_B10.log', level=logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.debug(device)



transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
    


train_set = datasets.ImageFolder("alphabet_train_large", transform = transformations)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)



test_set = datasets.ImageFolder("alphabet_test", transform = transformations)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=True)

classes = ('A', 'B', 'C', 'D', 'DEL', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'NOTHING', 'O', 'P', 'Q', 'R', 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
       


model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
model.eval()


# Replace default classifier with new classifier
model = nn.DataParallel(model)
model.to(device) #turning into GPU
logging.debug(f"Let's use {torch.cuda.device_count()} GPU")

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


for epoch in range(50):  # loop over the dataset multiple times
    logging.debug(f"epoch:{epoch}")
    running_loss = 0.0
    epoch_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        #for param in model.parameters():
            #print(param.data)
            
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        # print statistics
        running_loss += loss.item()
        epoch_loss += loss.item()
        
        if i % 10 == 9:    # print every 2000 mini-batches
            logging.debug('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
            
    writer.add_scalar('Loss/train',epoch_loss/len(train_set),epoch)
    

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network 
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    writer.add_scalar('Accuracy/test',100 * correct / total,epoch)
    PATH = f"./largeGoogLe_asl_net_SGD_B10_lr_0.001_{epoch}.pth"
    torch.save(model.state_dict(), PATH)
    
logging.debug('Finished Training')

PATH = './largeGoogLe_asl_net_SGD_B10_lr_0.001.pth'
torch.save(model.state_dict(), PATH)