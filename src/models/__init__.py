"""Model registry for the ear→heart ECG reconstruction models."""
from __future__ import annotations

from torch import nn

from .dcae import DCAE
from .deep_mf import DeepMF
from .deep_mf_mini import DeepMFMini
from .sdcae import SDCAE

MODEL_REGISTRY = {
    "sdcae": SDCAE,
    "dcae": DCAE,
    "deep_mf": DeepMF,
    "deep_mf_mini": DeepMFMini,
}


def build_model(name: str, in_channels: int = 1) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](in_channels)
