"""Fixed multi-filter front-end used by the Deep-MF family of models.

Deep-MF ("Deep Multi-Filter") does not take the raw 1-channel ear ECG directly;
it first splits it into 3 zero-phase Butterworth-filtered views (order 4):
band-pass 1-45 Hz, band-pass 1-5 Hz, and high-pass 1 Hz. This is a fixed
(non-learnable) preprocessing step, matching the original reference
implementation's `make_3_channel_ecg`.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt


class ThreeChannelFilterBank:
    """1-channel ear ECG -> 3-channel filtered ear ECG. Not learnable."""

    def __init__(self, fs: int = 250, order: int = 4):
        self.sos_bp_1_45 = butter(order, [1, 45], btype="bandpass", fs=fs, output="sos")
        self.sos_bp_1_5 = butter(order, [1, 5], btype="bandpass", fs=fs, output="sos")
        self.sos_hp_1 = butter(order, 1, btype="highpass", fs=fs, output="sos")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L) -> (B, 3, L), same device/dtype as x."""
        device, dtype = x.device, x.dtype
        x_np = x.detach().to("cpu", torch.float64).numpy()[:, 0, :]  # (B, L)
        ch1 = sosfiltfilt(self.sos_bp_1_45, x_np, axis=-1)
        ch2 = sosfiltfilt(self.sos_bp_1_5, x_np, axis=-1)
        ch3 = sosfiltfilt(self.sos_hp_1, x_np, axis=-1)
        out = np.stack([ch1, ch2, ch3], axis=1)  # (B, 3, L)
        return torch.from_numpy(out.copy()).to(device=device, dtype=dtype)
