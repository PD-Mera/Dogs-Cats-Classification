import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *

from config import *


class ClsfModel():
    def __init__(self):
        super(ClsfModel, self).__init__()
        pass
    def forward(self, x):
        return x


def init_model(train_config, load_checkpoint = None):
    assert train_config['modelname'] in MODEL_AVAILABLE, f'"modelname" in `config.py` must in {MODEL_AVAILABLE}'
    model_name = train_config['modelname']
    if model_name == 'custom':
        model = ClsfModel()
    else:
        if model_name == 'resnet18':
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
           
        elif model_name == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        elif model_name == 'resnet152':
            backbone = resnet152(weights=ResNet152_Weights.DEFAULT)

        model = nn.Sequential(
            backbone,
            nn.Linear(1000, CLASS_INFO['num']),
            nn.Softmax(dim=1)
        )
    
    
    if load_checkpoint is not None:
        model.load_state_dict(torch.load(load_checkpoint))
 
    return model