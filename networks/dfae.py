from torch import nn
from .padding_same_conv import Conv2d

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
            nn.Linear(1024 * 4 * 4, 1024),
            nn.Linear(1024, 1024 * 4 * 4),
            Reshape(),
            Conv2D_Transpose(1024, 512),
        )
        self.decoder_A = nn.Sequential(
            Conv2D_Transpose(512, 256),
            Conv2D_Transpose(256, 128),
            Conv2D_Transpose(128, 64),
            nn.Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )
        self.decoder_B = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),
            nn.Conv2d(64, 3, kernel_size=5, padding=1),
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


class Conv2D(nn.module):

    def __init__(self, cin, cout, kernel_size, stride):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=5, stride=2),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class Conv2D_Transpose:

    def __init__(self, cin, cout, kernel_size, stride):
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class Flatten(nn.Module):

    def forward(self, input):
        output = input.view(input.size(0), -1)
        return output


class Reshape(nn.Module):

    def forward(self, input):
        output = input.view(-1, 1024, 4, 4)  # channel * 4 * 4
        return output


class _ConvLayer(nn.Sequential):

    def __init__(self, input_features, output_features):
        super(_ConvLayer, self).__init__()
        self.add_module('conv2', Conv2d(
            input_features, 
            output_features,
            kernel_size=5, 
            stride=2)
        )
        self.add_module('leakyrelu', nn.LeakyReLU(0.1, inplace=True))


class _UpScale(nn.Sequential):

    def __init__(self, input_features, output_features):
        super(_UpScale, self).__init__()
        self.add_module('conv2_', Conv2d(
            input_features, 
            output_features * 4,
            kernel_size=3)
        )
        self.add_module('leakyrelu', nn.LeakyReLU(0.1, inplace=True))
        self.add_module('pixelshuffler', _PixelShuffler())


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
