"""Training loop, configurable losses, and evaluation metrics."""
from __future__ import annotations

from .train import TrainConfig, train_fold, evaluate, seed_everything
from .losses import (
    ECGFounderPerceptualLoss,
    FinalECGCombinedLoss,
    MSEPearsonLoss,
    PearsonCorrelationLoss,
    TotalVarianceLoss,
    DEFAULT_FOUNDER_CHECKPOINT,
    z_score_normalize_pt,
)
from .metrics import (
    batch_metrics,
    pqrst_pearson,
    pearson_per_window,
    average_dicts,
)

__all__ = [
    "TrainConfig",
    "train_fold",
    "evaluate",
    "seed_everything",
    "FinalECGCombinedLoss",
    "MSEPearsonLoss",
    "ECGFounderPerceptualLoss",
    "PearsonCorrelationLoss",
    "TotalVarianceLoss",
    "DEFAULT_FOUNDER_CHECKPOINT",
    "z_score_normalize_pt",
    "batch_metrics",
    "pqrst_pearson",
    "pearson_per_window",
    "average_dicts",
]
