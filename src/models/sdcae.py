"""SDCAE — Spiking Denoising Convolutional AutoEncoder (proposed model).

4-bit LSQ-quantized 1D conv/transposed-conv with integer MultiSpike activations.
"""
from __future__ import annotations

from torch import nn

from .layers import Conv1dLSQ, ConvTranspose1dLSQ, MultiSpike


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = Conv1dLSQ(in_channels, 8, kernel_size=201, stride=1, padding=100)
        self.conv2 = Conv1dLSQ(8, 8, kernel_size=51, stride=1, padding=25)
        self.conv3 = Conv1dLSQ(8, 8, kernel_size=51, stride=1, padding=25)
        self.conv4 = Conv1dLSQ(8, 8, kernel_size=51, stride=1, padding=25)
        self.bn1, self.bn2, self.bn3, self.bn4 = (nn.BatchNorm1d(8) for _ in range(4))
        self.spike1, self.spike2, self.spike3, self.spike4 = (MultiSpike() for _ in range(4))
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout1(self.spike1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.spike2(self.bn2(self.conv2(x))))
        x = self.spike3(self.bn3(self.conv3(x)))
        x = self.spike4(self.bn4(self.conv4(x)))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = ConvTranspose1dLSQ(8, 8, kernel_size=51, stride=1, padding=25)
        self.deconv2 = ConvTranspose1dLSQ(8, 8, kernel_size=51, stride=1, padding=25)
        self.deconv3 = ConvTranspose1dLSQ(8, 8, kernel_size=51, stride=1, padding=25)
        self.deconv4 = ConvTranspose1dLSQ(8, 1, kernel_size=201, stride=1, padding=100)
        self.bn1, self.bn2, self.bn3 = (nn.BatchNorm1d(8) for _ in range(3))
        self.spike1, self.spike2, self.spike3 = (MultiSpike() for _ in range(3))

    def forward(self, x):
        x = self.spike1(self.bn1(self.deconv1(x)))
        x = self.spike2(self.bn2(self.deconv2(x)))
        x = self.spike3(self.bn3(self.deconv3(x)))
        x = self.deconv4(x)
        return x


class SDCAE(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))
