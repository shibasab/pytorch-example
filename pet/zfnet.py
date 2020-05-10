import torch.nn as nn
import torch.nn.functional as F


class ZFNet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(ZFNet, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=96,
                      kernel_size=(7, 7), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256,
                      kernel_size=(5, 5), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384,
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256*7*7, out_features=4096),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x
