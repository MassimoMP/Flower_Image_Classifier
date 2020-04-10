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

#loadig the model function

def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    if model == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model == "densenet201":
        model = models.densenet201(pretrained=True)
    model.classifier = checkpoint["classifier"]
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

#processing the image for prediction

def process_image(image):
    im = Image.open(image)
    
    #Scaling the image
    #MAX_SIZE = (256, 256) 
    #im.thumbnail(MAX_SIZE)
    width, height = im.size 
    aspect_ratio = width/height 
    
    if min(width, height) == width:
        im.resize((int(256/aspect_ratio), 256))
    else:
        im.resize(((256),int(256*aspect_ratio)))
      
    #Croping out the center of the image
    width, height = im.size 
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    im_crop = im.crop((left, top, right, bottom))
    #Converting color channels values to 0-1
    np_image = np.array(im_crop)
    np_image = np_image/255
    #Normalizing the color channels values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image-mean)/std
    #transoposing the image for Pythorch
    final_image = np_image.transpose()
    #torch_image = torch.from_numpy(final_image)
    return final_image


#predicting the image function

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
     if torch.cuda.is_available():
        device = device
    else: 
        device = "cpu"
    image = process_image(image_path)
    #torch_img = torch.from_numpy(img).type(torch.FloatTensor)
    #model = load_checkpoint(saved_model, model_name)
    model.to(device)
    torch_img = torch.from_numpy(image).type(torch.FloatTensor)
    torch_img.unsqueeze_(0)
    torch_img = torch_img.to(device)
    model.eval()
    with torch.no_grad():
        logps = model.forward(torch_img)
        ps = torch.exp(logps)
        probs, indexes = ps.topk(topk, dim=1)
        indexes = indexes.to("cpu")
        probs = probs.to("cpu")
        model.class_to_idx
        idx_to_classes = {v:k for k,v in model.class_to_idx.items()}
        probs = probs.numpy()[0]
        classes = np.array([idx_to_classes[idx] for idx in indexes.numpy()[0]])
    
    return probs, classes, image_path
    