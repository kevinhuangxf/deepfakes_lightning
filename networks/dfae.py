from torch import nn
from networks.conv_block import Conv2D, Conv2D_Transpose, ResidualBlock

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
        )
        self.inter = nn.Sequential(
            nn.Linear(1024 * 8 * 8, 1024),
            nn.Linear(1024, 1024 * 8 * 8),
            Reshape(),
            Conv2D_Transpose(1024, 512),
        )
        self.decoder_src = nn.Sequential(
            Conv2D_Transpose(512, 256),
            ResidualBlock(256),
            Conv2D_Transpose(256, 128),
            ResidualBlock(128),
            Conv2D_Transpose(128, 64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )
        self.decoder_dst = nn.Sequential(
            Conv2D_Transpose(512, 256),
            ResidualBlock(256),
            Conv2D_Transpose(256, 128),
            ResidualBlock(128),
            Conv2D_Transpose(128, 64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x, label):
        out = self.encoder(x)
        out = self.inter(out)
        if label == 'src':
            out = self.decoder_src(out)
        if label == 'dst':
            out = self.decoder_dst(out)
        return out


class Flatten(nn.Module):

    def forward(self, input):
        output = input.view(input.size(0), -1)
        return output


class Reshape(nn.Module):

    def forward(self, input):
        output = input.view(-1, 1024, 8, 8)  # C * H * W
        return output
