import torch
from torch import nn


class Conv2D(nn.Module):

    def __init__(self, cin, cout, kernel_size=5, stride=2, padding=2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class Conv2D_Transpose(nn.Module):

    def __init__(self, cin, cout, kernel_size=5, stride=2, padding=2, output_padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)