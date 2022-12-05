CLASS_INFO = {
    'num': 2,
    'name': [
        'cat',
        'dog',
    ]
}

TRAINING_CFG = {
    'path': './data/train/',
    'class': CLASS_INFO,
    'image_size': (256, 256),
    'modelname': 'resnet18',
    'epoch': 10, 
    'batch_size': 128,
    'optimizer': 'Adam',
    'learning_rate': 1e-3,
    'loss': 'CE',
    'model_savepath': './weights/',
    'load_checkpoint': None,
}

VALID_CFG = {
    'path': './data/valid/',
    'class': CLASS_INFO,
    'image_size': (256, 256),
    'batch_size': 16,
}

MODEL_AVAILABLE = ['custom', 'resnet18', 'resnet50']
OPTIMIZER_AVAILABLE = ['Adam']
LOSS_AVAILABLE = ['custom', 'CE']
