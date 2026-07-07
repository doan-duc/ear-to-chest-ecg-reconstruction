"""Deep-MF-mini — smaller full-precision baseline (6 channels, kernels 201/51, BN).

Unlike the full Deep-MF, the reference implementation instantiates this variant
with `in_channels=1` (raw ear ECG) — no multi-filter front-end.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 6, kernel_size=201, stride=1, padding=25)
        self.conv2 = nn.Conv1d(6, 6, kernel_size=51, stride=1, padding=25)
        self.conv3 = nn.Conv1d(6, 6, kernel_size=51, stride=1, padding=25)
        self.conv4 = nn.Conv1d(6, 6, kernel_size=51, stride=1)
        self.bn1, self.bn2, self.bn3, self.bn4 = (nn.BatchNorm1d(6) for _ in range(4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(6, 6, kernel_size=51, stride=1)
        self.deconv2 = nn.ConvTranspose1d(6, 6, kernel_size=51, stride=1, padding=25)
        self.deconv3 = nn.ConvTranspose1d(6, 6, kernel_size=51, stride=1, padding=25)
        self.deconv4 = nn.ConvTranspose1d(6, 1, kernel_size=201, stride=1, padding=25)
        self.bn1, self.bn2, self.bn3 = (nn.BatchNorm1d(6) for _ in range(3))

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        return x


class DeepMFMini(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))
