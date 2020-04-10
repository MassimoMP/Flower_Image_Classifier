import torchvision.models as models
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
import time
#defining a function for each pre-train model that the user could choose. 
def vgg16(hddn1, hddn2):
    
    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

        model.classifier = nn.Sequential(nn.Linear(25088, hddn1),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(hddn1, hddn2),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(hddn2, 102),
                                         nn.LogSoftmax(dim=1))
        return model

def densenet201(hddn1, hddn2):
    
    model = models.densenet201(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

        model.classifier = nn.Sequential(nn.Linear(1920, hddn1),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(hddn1, hddn2),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(hddn2, 102),
                                         nn.LogSoftmax(dim=1))
        return model

def resnet101(hddn1, hddn2):
    
    model = models.resnet101(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

        model.classifier = nn.Sequential(nn.Linear(2048, hddn1),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(hddn1, hddn2),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(hddn2, 102),
                                         nn.LogSoftmax(dim=1))
        return model

#choosing the model
def choosing_the_model(model_name, hddn1, hddn2):
    if model_name == "vgg16":
        model = vgg16(hddn1, hddn2)
        return model
    elif model_name == "densenet201":
        model = densenet201(hddn1, hddn2)
        return model
    elif model_name == "resnet101":
        model = resnet101(hddn1, hddn2)
        return model