import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from config import *

def init_model(train_config, load_checkpoint = None):
    model_name = train_config['modelname']
    if model_name == 'resnet18':
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    if load_checkpoint is not None:
        backbone.load_state_dict(torch.load(load_checkpoint))

    model = nn.Sequential(
        backbone,
        nn.Linear(1000, CLASS_INFO['num']),
        nn.Softmax(dim=1)
    )
    
    return model