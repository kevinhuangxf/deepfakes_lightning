from torch import nn
from .conv_block import Conv2D, Conv2D_Transpose

## deep fakes auto-encoder
class DeepFakesAutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            Conv2D(3, 128),
            Conv2D(128, 256),
            Conv2D(256, 512),
            Conv2D(512, 1024),
            Flatten(),
            nn.Linear(1024 * 16 * 16, 1024),
            nn.Linear(1024, 1024 * 16 * 16),
            Reshape(),
            Conv2D_Transpose(1024, 512),
        )
        self.decoder_A = nn.Sequential(
            Conv2D_Transpose(512, 256),
            ResidualBlock(256),
            Conv2D_Transpose(256, 128),
            ResidualBlock(128),
            Conv2D_Transpose(128, 64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )
        self.decoder_B = nn.Sequential(
            Conv2D_Transpose(512, 256),
            ResidualBlock(256),
            Conv2D_Transpose(256, 128),
            ResidualBlock(128),
            Conv2D_Transpose(128, 64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x, select='A'):
        if select == 'A':
            out = self.encoder(x)
            out = self.decoder_A(out)
        else:
            out = self.encoder(x)
            out = self.decoder_B(out)
        return out


class Flatten(nn.Module):

    def forward(self, input):
        output = input.view(input.size(0), -1)
        return output


class Reshape(nn.Module):

    def forward(self, input):
        output = input.view(-1, 1024, 16, 16)  # channel * 4 * 4
        return output


class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel_size=5 ):
        super().__init__()
        self.conv1 = nn.Conv2d( ch, ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d( ch, ch, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, inp):
        x = self.conv1(inp)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = nn.LeakyReLU(0.2)(inp + x)
        return x


class _PixelShuffler(nn.Module):

    def forward(self, input):
        batch_size, c, h, w = input.size()
        rh, rw = (2, 2)
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)
        out = input.view(batch_size, rh, rw, oc, h, w)
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous()
        out = out.view(batch_size, oc, oh, ow)  # channel first
        return out
