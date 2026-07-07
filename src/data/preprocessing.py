"""Preprocessing: raw CSV → filtered, trimmed continuous signals.

Pipeline per subject:
  1. read 2-column CSV  (col 0 = ear ECG = input x, col 1 = chest ECG = target y)
  2. trim unstable start/end (variable per subject, keeps the most usable span)
  3. band-stop 49–51 Hz + band-pass 1–40 Hz (both channels)
  4. PQRST masks from the chest reference (Pan-Tompkins)

Windowing and z-scoring happen at dataset load time so that train and test
splits can use different window overlaps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .rpeak import build_pqrst_masks


@dataclass
class PreprocessConfig:
    fs: int = 250
    window_sec: float = 2.0          # 500 samples at 250 Hz
    train_overlap: float = 0.9       # dense windows for training (stride = 50)
    test_overlap: float = 0.5        # sparser windows for val/test (stride = 250)
    # Variable end trimming removes unstable boundaries while preserving usable signal.
    min_trim_sec: float = 5.0        # always drop at least this from each end
    env_win_sec: float = 1.0         # RMS-envelope window for stability check
    env_low: float = 0.25            # keep where envelope in [low, high] * median
    env_high: float = 4.0
    # Filtering.
    notch_low: float = 49.0
    notch_high: float = 51.0
    notch_order: int = 6
    band_low: float = 1.0
    band_high: float = 40.0
    band_order: int = 2


def read_ecg_csv(csv_path) -> tuple[np.ndarray, np.ndarray]:
    """Return (ear, chest) 1D float32 arrays from a 2-column CSV (no header)."""
    df = pd.read_csv(csv_path, header=None).apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path}: expected 2 columns, got {df.shape}")
    ear = df.iloc[:, 0].to_numpy(dtype=np.float32)
    chest = df.iloc[:, 1].to_numpy(dtype=np.float32)
    n = min(len(ear), len(chest))
    return ear[:n], chest[:n]


def _sos_filter(x, fs, band, order, btype):
    nyq = fs / 2.0
    sos = butter(order, [b / nyq for b in band], btype=btype, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def filter_ecg(x, cfg: PreprocessConfig) -> np.ndarray:
    x = _sos_filter(x, cfg.fs, (cfg.notch_low, cfg.notch_high), cfg.notch_order, "bandstop")
    x = _sos_filter(x, cfg.fs, (cfg.band_low, cfg.band_high), cfg.band_order, "bandpass")
    return x


def find_stable_span(ref: np.ndarray, cfg: PreprocessConfig) -> tuple[int, int]:
    """Largest contiguous span where the reference RMS envelope is physiological.

    Uses the chest reference to drop flat / saturated / motion-corrupted ends,
    variable per subject. Returns (start, end) sample indices.
    """
    fs = cfg.fs
    n = len(ref)
    w = max(1, int(cfg.env_win_sec * fs))
    r = ref - np.mean(ref)
    env = np.sqrt(np.convolve(r ** 2, np.ones(w) / w, mode="same"))
    med = np.median(env[env > 0]) if np.any(env > 0) else 0.0
    if med <= 0:
        return 0, n
    ok = (env >= cfg.env_low * med) & (env <= cfg.env_high * med)
    # Locate the longest stable contiguous span.
    best_s = best_e = 0
    s = None
    for i, v in enumerate(ok):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s > best_e - best_s:
                best_s, best_e = s, i
            s = None
    if s is not None and n - s > best_e - best_s:
        best_s, best_e = s, n
    # Enforce the minimum trim at both recording boundaries.
    m = int(cfg.min_trim_sec * fs)
    start = max(best_s, m)
    end = min(best_e, n - m)
    if end - start < int(cfg.window_sec * fs):
        # Fall back to the centrally trimmed span if the stability gate is too strict.
        start, end = m, n - m
    return start, end


def segment_windows(x: np.ndarray, cfg: PreprocessConfig, overlap: float,
                    axis: int = -1) -> np.ndarray:
    """Sliding windows. x: (..., T) → (..., n_windows, window_size)."""
    window_size = int(cfg.window_sec * cfg.fs)
    stride = max(1, int(window_size * (1 - overlap)))
    x = np.moveaxis(x, axis, -1)
    T = x.shape[-1]
    starts = range(0, T - window_size + 1, stride)
    windows = np.stack([x[..., s:s + window_size] for s in starts], axis=-2)
    return windows.astype(np.float32)


def zscore_per_window(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)


def window_subject(ear_f, chest_f, masks, cfg: PreprocessConfig, overlap: float):
    """Segment continuous signals into z-scored windows at the given overlap.

    Returns x (n_win, 500), y (n_win, 500), m (n_win, 4, 500).
    """
    x_w = segment_windows(ear_f, cfg, overlap)              # (n_win, 500)
    y_w = segment_windows(chest_f, cfg, overlap)            # (n_win, 500)
    m_w = segment_windows(masks, cfg, overlap, axis=-1)     # (4, n_win, 500)
    m_w = np.moveaxis(m_w, 0, 1)                            # (n_win, 4, 500)
    x_w = zscore_per_window(x_w)
    y_w = zscore_per_window(y_w)
    return x_w.astype(np.float32), y_w.astype(np.float32), m_w.astype(np.float32)


def preprocess_subject(csv_path, cfg: PreprocessConfig):
    """Full pipeline for one subject up to (but not including) windowing.

    Returns the continuous filtered + trimmed signals:
        ear (T,), chest (T,), masks (4, T).
    """
    ear, chest = read_ecg_csv(csv_path)
    ear_f = filter_ecg(ear, cfg)
    chest_f = filter_ecg(chest, cfg)
    start, end = find_stable_span(chest_f, cfg)
    ear_f, chest_f = ear_f[start:end], chest_f[start:end]

    masks, _ = build_pqrst_masks(chest_f[None, :], fs=cfg.fs)   # (1, 4, T)
    masks = masks[0]
    return ear_f.astype(np.float32), chest_f.astype(np.float32), masks.astype(np.float32)


def preprocess_all(raw_dir, cfg: PreprocessConfig):
    """Preprocess every ecg_data_*.csv. Returns dict subject_id -> (ear, chest, masks)."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("ecg_data_*.csv"))
    out = {}
    for f in files:
        sid = f.stem  # e.g. ecg_data_01
        out[sid] = preprocess_subject(f, cfg)
    return out
