import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)

        )


