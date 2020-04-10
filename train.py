import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torchvision.models as models
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
import time
import argparse
import model_functions 
from model_functions import choosing_the_model
#from model_functions import training_data


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Defining your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


#Load the datasets with ImageFolder

train_datasets = datasets.ImageFolder(train_dir, train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, test_transforms)

#Using the image datasets and the trainforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)

#function to train the model; user must input the the model, the epochs and the learn rate. 

def training_data(model, epochs, learn_rate, device): #check how to implement trainloader
    if torch.cuda.is_available():
        device = device
    else: 
        device = "cpu"
    model.to(device);
    start = time.time()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader: 
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
    
        else:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    valid_loss += criterion(logps, labels)
                    #valid_loss += batch_loss.item()
                
                    #accuracy calculation: 
                
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            model.train()
            print(f"Epoch {e+1}/{e}.. "
                  f"Train loss: {running_loss/len(trainloader):.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validloader):.3f}")
    print(f"Time: {(time.time() - start)/3:.3f} seconds")

#argparse used for command line user input
parser = argparse.ArgumentParser(description = "Giving neccesary inputs to train the network.")
parser.add_argument("-mn", "--model_name", type = str, choices = ['vgg16', 'densenet201', 'resnet101'], required = True, help = "Pick one between: vgg16, densenet201, resnet101.")
parser.add_argument("-hd1", "--hddn1", type = int, required = True, help = "Number hidden units on the first layer. Note: if vgg16 was chosen hd1 < 25088, if densenet201 was chosen hd1 < 1920, if resnet101 was chosen hd1 < 2048")
parser.add_argument("-hd2","--hddn2", type = int, required = True, help = "Number hidden units on the second layer. Note: hd2 < hd1")
#parser.add_argument("-drp","--dropout", type = int, required = True, help = "Dropout value to avoid overfitting; between 0 and 1.")

parser.add_argument("-e","--epochs", type = int, required = True, help = "Number of times to train the model.")
parser.add_argument("-lr","--learn_rate", type = float, required = True, help = "Learning rate of the model.")
parser.add_argument("-de", "--device", type = str, choices = ['cpu', 'cuda'], required = True, help = "Choose between gpu or cpu.")
args = parser.parse_args()

#running the needed functions to train the model
model = choosing_the_model(args.model_name, args.hddn1, args.hddn2) #args.dropout


training_data(model, args.epochs, args.learn_rate, args.device)

# saving the model as a checkpoint
model.class_to_idx = train_datasets.class_to_idx
checkpoint = {"class_to_idx": model.class_to_idx,
              "classifier": model.classifier,
              "state_dict": model.state_dict()}
             
torch.save(checkpoint, 'checkpoint.pth')