from mnist_dataloader import MnistDataset, ToTensor
from model import Net
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import csv


def save_csv(csv_file, inp):
    with open(csv_file, "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(inp)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = MnistDataset(
    "data/test.csv", transform=transforms.Compose([ToTensor()]),
    train=False)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

net = Net(in_channel=1, num_classes=10)
net.load_state_dict(torch.load("models/mnist_model4.pkl"))
net = net.to(device)
net.eval()

out_list = [["ImageId", "Label"]]

with torch.no_grad():
    for i, sample in enumerate(test_dataloader, 0):
        image = sample['image'].float().to(device)
        output = net(image)
        _, predicted = torch.max(output.data, 1)
        out_list.append([i+1, predicted[0].item()])

save_csv("results/mnist_result4.csv", out_list)
print("\nFinished!")
