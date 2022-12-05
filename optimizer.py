import torch

from config import *


class ClsfLoss():
    def __init__(self):
        super(ClsfLoss, self).__init__()
        pass
    def forward(self, x):
        return x


def init_optimizer(model, train_config):
    assert train_config['optimizer'] in OPTIMIZER_AVAILABLE, f'"optimizer" in `config.py` must in {OPTIMIZER_AVAILABLE}'
    if train_config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])

    return optimizer
