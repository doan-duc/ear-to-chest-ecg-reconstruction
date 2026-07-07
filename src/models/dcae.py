"""DCAE — full-precision denoising conv autoencoder baseline (4->8->16->32)."""
from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 4, kernel_size=75, stride=1, padding=37)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=45, stride=1, padding=22)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=45, stride=1, padding=22)
        self.conv4 = nn.Conv1d(16, 32, kernel_size=45, stride=1, padding=22)
        self.bn1, self.bn2 = nn.BatchNorm1d(4), nn.BatchNorm1d(8)
        self.bn3, self.bn4 = nn.BatchNorm1d(16), nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2(F.relu(self.bn3(self.conv3(x))))
        x = self.drop2(F.relu(self.bn4(self.conv4(x))))
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(32, 16, kernel_size=45, stride=1, padding=22)
        self.deconv2 = nn.ConvTranspose1d(16, 8, kernel_size=45, stride=1, padding=22)
        self.deconv3 = nn.ConvTranspose1d(8, 4, kernel_size=45, stride=1, padding=22)
        self.deconv4 = nn.ConvTranspose1d(4, out_channels, kernel_size=75, stride=1, padding=37)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(16), nn.BatchNorm1d(8), nn.BatchNorm1d(4)

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        return x


class DCAE(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels=1)

    def forward(self, x):
        return self.decoder(self.encoder(x))
