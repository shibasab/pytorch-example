from cifar10_loader import load_cifer10
from lenet import LeNet

import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(dataloader, model, optimizer, criterion, batch_size, device, total_epoch, epoch):
    model.train()
    losses = []

    for i, sample in enumerate(dataloader, 0):
        optimizer.zero_grad()
        images, labels = sample
        images = images.to(device)
        labels = labels.view(batch_size, -1)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print('epoch: {} / {}, loss: {}'.format(epoch, total_epoch, np.mean(losses)))


def val(dataloader, trained_model, device, data_num):
    trained_model.eval()
    correct = 0
    accuracy = 0.0

    with torch.no_grad():
        for i, sample in enumerate(dataloader, 0):
            images, labels = sample
            images = images.to(device)
            labels = labels.view(batch_size, -1)
            labels = labels.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(
                labels.view(labels.size(0))).sum().item()
        accuracy = 100. * correct / data_num

    print('\nAccuracy: {}/{} ({:.4f}%)\n'.format(correct,
                                                 data_num, accuracy))
    return accuracy


if __name__ == '__main__':
    epochs = 50
    batch_size = 10
    best_acc = 0.0

    torch.manual_seed(1)

    _, test_dataset, train_loader, test_loader = load_cifer10(
        batch_size=batch_size, num_workers=6, data_dir='~/data', download=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LeNet(in_channel=3, num_classes=len(classes))
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(1, epochs+1):
        train(train_loader, model, optimizer, criterion,
              batch_size, device, epochs, epoch)
        acc = val(test_loader, model, device, len(test_dataset))

        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "models/cifar10_lenet.pkl")
    print('finished!')
