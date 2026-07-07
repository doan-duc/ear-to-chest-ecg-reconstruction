"""Deep-MF — full-precision baseline (6 channels, kernels 200/50, 'same' padding).

"MF" = multi-filter: the raw 1-channel ear ECG is first split into 3 fixed
Butterworth-filtered views (see filters.ThreeChannelFilterBank) before the
learnable encoder-decoder, matching the reference implementation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .filters import ThreeChannelFilterBank


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 6, kernel_size=200, stride=1, padding="same")
        self.conv2 = nn.Conv1d(6, 6, kernel_size=50, stride=1, padding="same")
        self.conv3 = nn.Conv1d(6, 6, kernel_size=50, stride=1, padding="same")
        self.conv4 = nn.Conv1d(6, 6, kernel_size=50, stride=1, padding="same")
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        x = self.dropout(torch.sigmoid(self.conv4(x)))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(6, 6, kernel_size=50, stride=1, padding=24)
        self.deconv2 = nn.ConvTranspose1d(6, 6, kernel_size=50, stride=1, padding=25)
        self.deconv3 = nn.ConvTranspose1d(6, 6, kernel_size=50, stride=1, padding=25)
        self.deconv4 = nn.ConvTranspose1d(6, 1, kernel_size=200, stride=1, padding=99)

    def forward(self, x):
        x = torch.sigmoid(self.deconv1(x))
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x


class DeepMF(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        if in_channels != 1:
            raise ValueError("DeepMF expects the raw 1-channel ear ECG; it builds "
                             "its own 3 filtered channels internally.")
        self.filter_bank = ThreeChannelFilterBank()
        self.encoder = Encoder(3)
        self.decoder = Decoder()

    def forward(self, x):
        x3 = self.filter_bank(x)
        return self.decoder(self.encoder(x3))
