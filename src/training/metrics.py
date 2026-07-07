"""Evaluation metrics for ECG reconstruction.

The primary metric is the per-window Pearson correlation over the PQRST complex;
we also report full-window Pearson, MSE, RMSE, SNR, R2 and PRD.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..data.masks import build_pqrst_complex_mask


def pearson_per_window(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-window Pearson r; pred/target: (B, 1, L) or (B, L). Returns (B,)."""
    p = pred.flatten(start_dim=1)
    t = target.flatten(start_dim=1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p ** 2).sum(dim=1) + eps) * torch.sqrt((t ** 2).sum(dim=1) + eps)
    return num / den


def pqrst_pearson(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  eps: float = 1e-8, reduce: bool = True):
    """Pearson over the contiguous PQRST complex, per window then averaged."""
    cm = build_pqrst_complex_mask(mask).to(dtype=pred.dtype)
    p = pred[:, 0, :] if pred.dim() == 3 else pred
    t = target[:, 0, :] if target.dim() == 3 else target
    denom = cm.sum(dim=1)
    valid = denom > 1
    pm = (p * cm).sum(1, keepdim=True) / (denom.unsqueeze(1) + eps)
    tm = (t * cm).sum(1, keepdim=True) / (denom.unsqueeze(1) + eps)
    pc = (p - pm) * cm
    tc = (t - tm) * cm
    num = (pc * tc).sum(1)
    den = torch.sqrt((pc ** 2).sum(1) * (tc ** 2).sum(1) + eps)
    r = num / den
    r = r[valid & torch.isfinite(r)]
    if not reduce:
        return r
    return r.mean() if r.numel() > 0 else pred.new_tensor(0.0)


def batch_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    err = pred - target
    mse = torch.mean(err ** 2)
    rmse = torch.sqrt(mse)
    pear = pearson_per_window(pred, target).mean()
    signal_power = torch.mean(target ** 2)
    noise_power = torch.mean(err ** 2) + 1e-8
    snr = 10.0 * torch.log10(signal_power / noise_power)
    target_var = torch.sum((target - target.mean()) ** 2) + 1e-8
    r2 = 1.0 - torch.sum(err ** 2) / target_var
    prd = 100.0 * torch.sqrt(torch.sum(err ** 2) / (torch.sum(target ** 2) + 1e-8))
    out = {
        "mse": float(mse), "rmse": float(rmse), "pearson": float(pear),
        "snr": float(snr), "r2": float(r2), "prd": float(prd),
    }
    if mask is not None:
        out["pqrst_pearson"] = float(pqrst_pearson(pred, target, mask))
    else:
        out["pqrst_pearson"] = out["pearson"]
    return out


class StreamingMetrics:
    """Accumulate evaluation metrics without overweighting the final small batch."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.total_samples = 0
        self.err_sq_sum = 0.0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.pearson_sum = 0.0
        self.pearson_count = 0
        self.pqrst_sum = 0.0
        self.pqrst_count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor | None = None) -> None:
        pred = pred.detach().float()
        target = target.detach().float()
        err = pred - target

        self.total_samples += int(target.numel())
        self.err_sq_sum += float(torch.sum(err ** 2).cpu())
        self.target_sum += float(torch.sum(target).cpu())
        self.target_sq_sum += float(torch.sum(target ** 2).cpu())

        pear = pearson_per_window(pred, target)
        pear = pear[torch.isfinite(pear)]
        self.pearson_sum += float(torch.sum(pear).cpu())
        self.pearson_count += int(pear.numel())

        if mask is None:
            pqrst = pear
        else:
            pqrst = pqrst_pearson(pred, target, mask.detach().float(), reduce=False)
            pqrst = pqrst[torch.isfinite(pqrst)]
        self.pqrst_sum += float(torch.sum(pqrst).cpu())
        self.pqrst_count += int(pqrst.numel())

    def compute(self) -> dict:
        if self.total_samples == 0:
            return {}

        mse = self.err_sq_sum / self.total_samples
        rmse = math.sqrt(mse)
        signal_power = self.target_sq_sum / self.total_samples
        noise_power = mse + self.eps
        snr = 10.0 * math.log10(max(signal_power, self.eps) / noise_power)
        target_var = self.target_sq_sum - (self.target_sum ** 2 / self.total_samples)
        r2 = 1.0 - self.err_sq_sum / (target_var + self.eps)
        prd = 100.0 * math.sqrt(self.err_sq_sum / (self.target_sq_sum + self.eps))

        pearson = self.pearson_sum / self.pearson_count if self.pearson_count else 0.0
        pqrst = self.pqrst_sum / self.pqrst_count if self.pqrst_count else pearson
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "pearson": float(pearson),
            "snr": float(snr),
            "r2": float(r2),
            "prd": float(prd),
            "pqrst_pearson": float(pqrst),
            "n_windows": int(self.pearson_count),
            "n_pqrst_windows": int(self.pqrst_count),
        }


def average_dicts(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
