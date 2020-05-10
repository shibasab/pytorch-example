from dataloader import PetDataset, ToTensor
from network_in_network import NIN

import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


def train(dataloader, model, optimizer, criterion, batch_size, device, total_epoch, epoch):
    model.train()
    losses = []

    for i, sample in enumerate(dataloader, 0):
        optimizer.zero_grad()
        image, label = sample['image'], sample['label']
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        loss = criterion(output, label)
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
            image, label = sample['image'], sample['label']
            image = image.to(device)
            label = label.to(device)
            output = trained_model(image)
            _, predicted = torch.max(output.data, 1)
            correct += predicted.eq(label).sum().item()
        accuracy = 100. * correct / data_num

    print('\nAccuracy: {}/{} ({:.4f}%)\n'.format(correct,
                                                 data_num, accuracy))
    return accuracy


if __name__ == '__main__':
    epochs = 50
    batch_size = 10
    best_acc = 0.0

    classes = {'Abyssinian': 0, 'american bulldog': 1, 'american pit bull terrier': 2, 'basset hound': 3, 'beagle': 4, 'Bengal': 5, 'Birman': 6, 'Bombay': 7, 'boxer': 8, 'British Shorthair': 9, 'chihuahua': 10, 'Egyptian Mau': 11, 'english cocker spaniel': 12, 'english setter': 13, 'german shorthaired': 14, 'great pyrenees': 15, 'havanese': 16, 'japanese chin': 17,
               'keeshond': 18, 'leonberger': 19, 'Maine Coon': 20, 'miniature pinscher': 21, 'newfoundland': 22, 'Persian': 23, 'pomeranian': 24, 'pug': 25, 'Ragdoll': 26, 'Russian Blue': 27, 'saint bernard': 28, 'samoyed': 29, 'scottish terrier': 30, 'shiba inu': 31, 'Siamese': 32, 'Sphynx': 33, 'staffordshire bull terrier': 34, 'wheaten terrier': 35, 'yorkshire terrier': 36}

    torch.manual_seed(1)

    img_dir = "..\\data\\oxford-pet-dataset\\images"
    base = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(base, img_dir)
    dataset = PetDataset(img_dir=img_dir, transform=transforms.Compose([
        ToTensor()
    ]))

    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NIN(in_channel=3, num_classes=len(classes))
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(1, epochs+1):
        train(train_loader, model, optimizer, criterion,
              batch_size, device, epochs, epoch)
        acc = val(val_loader, model, device, len(val_dataset))

        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "models/pet_nin.pkl")
    print('finished!')
