import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MnistDataset(Dataset):
    def __init__(self, csv_file, transform=None, train=True):
        self.frame = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.train:
            image = self.frame.iloc[idx, 1:]
        else:
            image = self.frame.iloc[idx, :]
        image = np.array([image]).astype('float').reshape(28, 28, 1)
        image /= 255.0
        if self.train:
            index = self.frame.iloc[idx, 0]
            label = np.zeros(10)
            label[index] = 1
        if self.train:
            sample = {'image': image, 'label': label}
        else:
            sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image.sub_(self.mean[:, None]).div_(
            self.std[:, None])
        return {'image': image, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        if 'label' in sample:
            image, label = sample['image'], sample['label']
            image = image.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}
        else:
            image = sample['image']
            image = image.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image)}


def show_batch(sample_batched):
    images_batch, labels_batch = sample_batched['image'], sample_batched['labels']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == '__main__':
    mnist_dataset = MnistDataset(csv_file='data/train.csv',
                                 transform=transforms.Compose([
                                     ToTensor()
                                 ]))
    dataloader = DataLoader(mnist_dataset,
                            batch_size=4, shuffle=True, num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['label'].size())

        if i_batch == 3:
            print(sample_batched['image'])
            break
