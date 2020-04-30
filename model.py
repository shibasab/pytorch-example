import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(Net, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.block1 = nn.Sequential(
            nn.Conv2d(self.in_channel, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),
            nn.BatchNorm2d(16),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),
            nn.BatchNorm2d(32)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*26*26, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), 32 * 26 * 26)
        x = self.fc(x)
        return x
