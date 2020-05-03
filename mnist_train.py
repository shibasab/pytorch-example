from model import Net
from mnist_dataloader import MnistDataset, ToTensor
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


def train(dataloader, model, optimizer, criterion, batch_size, device, total_epoch, epoch):
    model.train()

    for i, sample in enumerate(dataloader, 0):
        optimizer.zero_grad()
        images = sample['image'].float()
        images = images.to(device)
        labels = sample['label'].view(batch_size, -1).long()
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        if i % 100 == 99:
            print('[{:d}, {:5d}] loss: {:.3f}'
                  .format(epoch, total_epoch, loss.item()))


def val(dataloader, trained_model, device, data_num):
    trained_model.eval()
    correct = 0
    accuracy = 0.0

    with torch.no_grad():
        for i, sample in enumerate(dataloader, 0):
            images = sample['image']
            images = images.to(device)
            labels = sample['label'].view(batch_size, -1)
            labels = labels.to(device)
            outputs = trained_model(images)
            labels = torch.where(labels == 1)[1]
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(
                labels.data.view_as(predicted)).sum().item()
            accuracy = 100. * correct / data_num

    print('\nAccuracy: {}/{} ({:.4f}%)\n'.format(correct,
                                                 data_num, accuracy))
    return accuracy


if __name__ == '__main__':
    epochs = 50
    batch_size = 100
    best_acc = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mnist_dataset = MnistDataset(csv_file='data/train.csv',
                                 transform=transforms.Compose([ToTensor()]),
                                 train=True)
    n_samples = len(mnist_dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        mnist_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True, num_workers=6)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size, shuffle=False)
    net = Net(in_channel=1, num_classes=10)
    net = net.to(device)

    best_model_wts = copy.deepcopy(net.state_dict())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(
        net.parameters(),  lr=0.001, alpha=0.9, eps=1e-08, weight_decay=0.0)

    for epoch in range(epochs):
        train(train_dataloader, net, optimizer, criterion,
              batch_size, device, epochs, epoch)
        acc = val(val_dataloader, net, device, len(val_dataset))

        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_model_wts)
    torch.save(net.state_dict(), "models/mnist_model4.pkl")
    print('finished!')
