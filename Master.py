

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(filename_suffix='large_Basic00001')
import logging
logging.basicConfig(filename='large_Basic00001.log', level=logging.DEBUG)


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
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)


test_set = datasets.ImageFolder("alphabet_test", transform = transformations)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

classes = ('A', 'B', 'C', 'D', 'DEL', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'NOTHING', 'O', 'P', 'Q', 'R', 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
       

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 29)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()



import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

logging.debug("STARTING TO TRAIN ")

logging.debug(f"len(train_set):{len(train_set)}")

for epoch in range(50):  # loop over the dataset multiple times
    logging.debug(f"epoch:{epoch}")
    epoch_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs,labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_loss += loss.item()
        
        if i % 30 == 29:   
            logging.debug('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 30))
            running_loss = 0.0
            
        writer.add_scalar('Loss/train',epoch_loss/len(train_set),epoch)
        
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                # calculate outputs by running images through the network 
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        writer.add_scalar('Accuracy/test',100 * correct / total,epoch)
        PATH = f"./large_asl_Basic_SGDlr_00001_{epoch}.pth"
        torch.save(net.state_dict(), PATH)
              

logging.debug('Finished Training')

PATH = './large_asl_Basic_SGDlr_00001.pth'
torch.save(net.state_dict(), PATH)