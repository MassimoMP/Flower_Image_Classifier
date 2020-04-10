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
import json
import argparse
import model_functions
from utility import load_checkpoint 
from utility import process_image
from utility import predict
import utility


parser = argparse.ArgumentParser(description = "Giving neccesary inputs to predict the flower image.")
parser.add_argument("-im","--image_path", type = str, help = "Image file Path; ex. flowers/test/27/image_06864.jpg.")
parser.add_argument("-k", "--topk", type = int, help = "The number of maximum probability outcome.")
parser.add_argument("-d", "--device", type = str, choices = ['cpu', 'cuda'], help = "Choose between gpu or cuda.")
parser.add_argument("-mn", "--model_name", type = str, choices = ['vgg16', 'densenet201', 'resnet101'], help = "Same model that was trained.")
args = parser.parse_args()

#importing load a JSON file
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
checkpoint = 'checkpoint.pth'

#loading the model
model = load_checkpoint(checkpoint, args.model_name)

#predicting the image input
probs, classes, image_path = predict(args.image_path, model, args.topk, args.device)

#getting the classes into the flower names
names = [cat_to_name[i] for i in classes]
#printing the results
print(f"The probabilities are: {probs}" )
print(f"The correspoonding flower names are: {names}")