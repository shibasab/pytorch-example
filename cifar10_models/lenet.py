import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(LeNet, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=6,
                      kernel_size=(5, 5), stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=6, out_channels=16,
                      kernel_size=(5, 5), stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Linear(in_features=120, out_features=64),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = x.view(x.size(0), 16 * 5 * 5)
        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x
