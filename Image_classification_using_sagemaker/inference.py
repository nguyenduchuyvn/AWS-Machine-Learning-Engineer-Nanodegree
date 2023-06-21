import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms

from PIL import Image
import io
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def net(num_classes = 133):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
   
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def model_fn(model_dir):   
    logger.info("model_dir: {} \n".format(model_dir))    
    model = net()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))    

    return model.to("cpu").eval()
    

def predict_fn(input_object, model):
    # Apply the necessary transformations on the input image
    transform = transforms.Compose([transforms.Resize(224),
                                    # transforms.CenterCrop(224),                         
                                    transforms.ToTensor()])
    input_object = transform(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction

# Preprocesses the input data and returns it as a PIL image object
def input_fn(request_body, content_type = 'image/jpeg'):
    if content_type == 'image/jpeg':
        return Image.open(io.BytesIO(request_body))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))
