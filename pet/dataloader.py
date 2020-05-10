import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class PetDataset(Dataset):
    def __init__(self, img_dir, img_size=224, transform=None, train=True):
        self.img_dir = img_dir
        self.files = os.listdir(img_dir)
        self.img_size = img_size
        self.classes = {'Abyssinian': 0, 'american bulldog': 1, 'american pit bull terrier': 2, 'basset hound': 3, 'beagle': 4, 'Bengal': 5, 'Birman': 6, 'Bombay': 7, 'boxer': 8, 'British Shorthair': 9, 'chihuahua': 10, 'Egyptian Mau': 11, 'english cocker spaniel': 12, 'english setter': 13, 'german shorthaired': 14, 'great pyrenees': 15, 'havanese': 16, 'japanese chin': 17,
                        'keeshond': 18, 'leonberger': 19, 'Maine Coon': 20, 'miniature pinscher': 21, 'newfoundland': 22, 'Persian': 23, 'pomeranian': 24, 'pug': 25, 'Ragdoll': 26, 'Russian Blue': 27, 'saint bernard': 28, 'samoyed': 29, 'scottish terrier': 30, 'shiba inu': 31, 'Siamese': 32, 'Sphynx': 33, 'staffordshire bull terrier': 34, 'wheaten terrier': 35, 'yorkshire terrier': 36}
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.files[idx])
        image = cv2.imread(img_name)
        image = cv2.resize(image, (self.img_size, self.img_size))
        class_name = ' '.join(self.files[idx].split('_')[:-1])
        label = np.array(self.classes[class_name])

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(), 'label': torch.from_numpy(label).long()}


if __name__ == '__main__':
    img_dir = "../data/oxford-pet-dataset/images"
    dataset = PetDataset(img_dir=img_dir, transform=transforms.Compose([
        ToTensor()
    ]))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=6)

    for i, sample in enumerate(dataloader):
        print(i, sample['image'].size(), sample['label'].size())

        if i == 3:
            print(sample['image'], sample['label'])
            break
