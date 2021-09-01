import torch
from torch import nn


def subnet_conv_1x1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256, 1), nn.ReLU(), nn.Conv2d(256, c_out, 1))


def subnet_conv(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, 256, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, c_out, 3, padding=1),
    )


class Conv2dSameSize(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size):
        same_padding = (
            kernel_size // 2
        )  # Padding that would keep the spatial dims the same
        super().__init__(in_channels, out_channels, kernel_size, padding=same_padding)


class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            Conv2dSameSize(in_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, out_channels, kernel_size=3),
        )

    def forward(self, inputs, context=None):
        return self.net.forward(inputs)
