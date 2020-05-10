import torch.nn as nn
import torch.nn.functional as F


class NIN(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(NIN, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=192,
                      kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=160,
                      kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=160, out_channels=96,
                      kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0),
            nn.Dropout2d()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=192,
                      kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0),
            nn.Dropout2d()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=(1, 1), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=self.num_classes,
                      kernel_size=(1, 1), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view((x.size(0), -1))
        x = F.softmax(x, dim=1)

        return x
