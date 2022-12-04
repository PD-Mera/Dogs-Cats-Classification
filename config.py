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
    'image_size': 256,
    'modelname': 'resnet18',
    'epoch': 10, 
    'batch_size': 128,
    'model_savepath': './weights/'
}

VALID_CFG = {
    'path': './data/valid/',
    'class': CLASS_INFO,
    'image_size': 256,
    'batch_size': 16,
}

TEST_CFG = {
    'path': './data/test/',
    'class': CLASS_INFO,
    'image_size': 256,
    'batch_size': 16,
}