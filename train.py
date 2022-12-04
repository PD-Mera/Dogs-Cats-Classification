import os
import torch
from torch.utils.data import DataLoader

from data import LoadDataset
from config import *
from model import init_model


def train(train_config, valid_config, test_config):
    model = init_model(train_config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_data = LoadDataset(train_config)
    valid_data = LoadDataset(valid_config)
    test_data = LoadDataset(test_config)
    train_loader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=valid_config['batch_size'])
    test_loader = DataLoader(test_data, batch_size=test_config['batch_size'])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    highest_acc = 0
    for epoch in range(train_config['epoch']):
        # Start Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        # Start Validating
        correct = 0
        for (data, target) in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        print(str(correct) + '/' + str(len(valid_loader.dataset)))
        accuracy = 100. * correct / len(valid_loader.dataset)
        print('\nValid set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(valid_loader.dataset), accuracy))

        if accuracy >= highest_acc:
            highest_acc == accuracy
            torch.save(model, os.path.join(train_config['model_savepath'] + f'training_epoch_{epoch}.pth'))


if __name__ == '__main__':
    train(train_config = TRAINING_CFG,
          valid_config=VALID_CFG,
          test_config=TEST_CFG)