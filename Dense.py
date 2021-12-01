import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(filename_suffix='large_dense_0001')
import logging
logging.basicConfig(filename='Large_dense_0001_B10.log', level=logging.DEBUG)

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
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False)

classes = ('A', 'B', 'C', 'D', 'DEL', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'NOTHING', 'O', 'P', 'Q', 'R', 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
       
net = models.densenet161(pretrained=True)
# Turn off training for their parameters
for param in net.parameters():
    param.requires_grad = False
    
classifier_input = net.classifier.in_features
num_labels = 29
classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))

# Replace default classifier with new classifier
net.classifier = classifier
net = nn.DataParallel(net)
net.to(device) #turning into GPU
logging.debug(f"Let's use {torch.cuda.device_count()} GPU")



import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

logging.debug("STARTING TO TRAIN ")

logging.debug(f"len(train_set):{len(train_set)}")

for epoch in range(50):  # loop over the dataset multiple times
    logging.debug(f"epoch:{epoch}")
    epoch_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

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
        
        if i % 10 == 9:   
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
                outputs = net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
               
        writer.add_scalar('Accuracy/test',100 * correct / total,epoch)
        PATH = f"./Large_dense_SGD_lr_0001_{epoch}.pth"
        torch.save(net.state_dict(), PATH)
              

logging.debug('Finished Training')

PATH = './Large_dense_SGD_lr_0001.pth'
torch.save(net.state_dict(), PATH)