import torch.nn as nn


class InputCNN(nn.Module):

    def __init__(self, in_channels, out_channels=64, mid_channels=32):
        super(InputCNN, self).__init__()

        self.n_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, out_channels, mid_channels)
        self.down1 = Down(64, 128)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        return x2


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
