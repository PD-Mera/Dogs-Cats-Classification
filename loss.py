import torch.nn as nn


class ClsfLoss():
    def __init__(self):
        super(ClsfLoss, self).__init__()
        pass
    def forward(self, x):
        return x


def init_loss(training_config):
    if training_config['loss'] == 'custom':
        loss = ClsfLoss()
    elif training_config['loss'] == 'CE':
        loss = nn.CrossEntropyLoss()
    return loss