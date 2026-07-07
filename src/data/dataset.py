"""Dataset assembly and leave-one-subject-out (LOSO) splitting.

The cache holds each subject's *continuous* filtered + trimmed signals (one .npz
each) so preprocessing only runs once. Windowing and z-scoring happen at load
time, which lets the train/validation splits use a dense overlap while the test
split uses a sparser overlap. Splitting into folds also happens in code — no
pre-split files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .preprocessing import PreprocessConfig, preprocess_subject, window_subject


def build_cache(raw_dir, cache_dir, cfg: PreprocessConfig, force: bool = False) -> list[str]:
    """Preprocess each CSV and cache continuous signals to cache_dir/<subject>.npz."""
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    subjects = []
    for f in sorted(raw_dir.glob("ecg_data_*.csv")):
        sid = f.stem
        subjects.append(sid)
        out = cache_dir / f"{sid}.npz"
        if out.exists() and not force:
            continue
        ear, chest, masks = preprocess_subject(f, cfg)
        np.savez_compressed(out, ear=ear, chest=chest, masks=masks)
    return subjects


def load_subject(cache_dir, sid: str):
    """Return the continuous (ear, chest, masks) arrays cached for a subject."""
    d = np.load(Path(cache_dir) / f"{sid}.npz")
    return d["ear"], d["chest"], d["masks"]


def loso_splits(subjects: list[str]) -> list[tuple[str, list[str]]]:
    """Leave-one-subject-out: returns [(test_subject, [train_subjects...]), ...]."""
    return [(s, [o for o in subjects if o != s]) for s in subjects]


def _stack(cache_dir, sids: list[str], cfg: PreprocessConfig, overlap: float):
    """Window every subject at the given overlap and concatenate."""
    xs, ys, ms = [], [], []
    for sid in sids:
        ear, chest, masks = load_subject(cache_dir, sid)
        x, y, m = window_subject(ear, chest, masks, cfg, overlap)
        xs.append(x); ys.append(y); ms.append(m)
    x = np.concatenate(xs, 0)[:, None, :]
    y = np.concatenate(ys, 0)[:, None, :]
    m = np.concatenate(ms, 0)
    return x.astype(np.float32), y.astype(np.float32), m.astype(np.float32)


def make_loaders(cache_dir, train_sids, test_sid, cfg: PreprocessConfig,
                 batch_size=128, val_sid=None, num_workers=0, seed=42):
    """Build train/val/test loaders.

    Train subjects are windowed at ``cfg.train_overlap``; the validation subject
    at ``cfg.val_overlap``; and the test subject at ``cfg.test_overlap``. If
    val_sid is None, one train subject is held out.
    """
    if val_sid is None:
        rng = np.random.default_rng(seed)
        val_sid = train_sids[int(rng.integers(len(train_sids)))]
    train_only = [s for s in train_sids if s != val_sid]

    def loader(sids, overlap, shuffle, drop_last=False):
        x, y, m = _stack(cache_dir, sids, cfg, overlap)
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return (loader(train_only, cfg.train_overlap, True, True),
            loader([val_sid], cfg.val_overlap, False),
            loader([test_sid], cfg.test_overlap, False),
            {"train_subjects": train_only, "val_subject": val_sid, "test_subject": test_sid})
