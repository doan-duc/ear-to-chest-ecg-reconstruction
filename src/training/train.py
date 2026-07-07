"""Training loop for one LOSO fold (unified protocol across all models)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .losses import DEFAULT_FOUNDER_CHECKPOINT, FinalECGCombinedLoss
from .metrics import StreamingMetrics
from ..models import build_model


@dataclass
class TrainConfig:
    epochs: int = 200
    patience: int = 25
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.02
    amp: bool = True
    grad_clip: float = 5.0
    seed: int = 42
    lambda_: float = 1.0
    beta_: float = 10.0
    gamma_: float = 5e-8
    alpha_: float = 0.0
    other_loss: int = 0
    founder_checkpoint: str = str(DEFAULT_FOUNDER_CHECKPOINT)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cosine_lr(epoch: int, cfg: TrainConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / max(1, cfg.warmup_epochs)
    progress = min(1.0, (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cos)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    metrics = StreamingMetrics()
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            pred = model(x)
        metrics.update(pred.float(), y.float(), m.float())
    return metrics.compute()


def train_fold(model_name, train_loader, val_loader, test_loader, cfg: TrainConfig,
               device=None, log_every=10, log_fn=print, on_epoch=None, criterion=None):
    """Train one fold. `on_epoch(epoch, val_pqrst_pearson)` is called after every
    epoch's validation (used by the Optuna tuner for pruning); it may raise to
    abort training early. `criterion` overrides the default FinalECGCombinedLoss
    (must return a tuple whose element [0] is the scalar total loss)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    lr = cfg.lr if model_name == "sdcae" else max(cfg.lr, 1e-3)
    model = build_model(model_name, 1).to(device)
    if criterion is None:
        criterion = FinalECGCombinedLoss(
            checkpoint_path=cfg.founder_checkpoint,
            device=device,
            lambda_=cfg.lambda_,
            beta_=cfg.beta_,
            gamma_=cfg.gamma_,
            alpha_=cfg.alpha_,
            other_loss=cfg.other_loss,
        ).to(device)
    else:
        criterion = criterion.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    best_metric, best_state, bad = -1.0, None, 0
    for epoch in range(cfg.epochs):
        cur_lr = _cosine_lr(epoch, cfg) * (lr / cfg.lr)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        model.train()
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                pred = model(x)
                total = criterion(pred, y)[0]
            scaler.scale(total).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

        val = evaluate(model, val_loader, device, cfg.amp)
        vm = float(val.get("pqrst_pearson", -1.0))
        improved = vm > best_metric + 1e-5
        if improved:
            best_metric, bad = vm, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if (epoch + 1) % log_every == 0 or improved:
            log_fn(f"[{model_name}] epoch {epoch+1}/{cfg.epochs} lr={cur_lr:.2e} "
                   f"val_pqrst={vm:.5f} best={best_metric:.5f} bad={bad}/{cfg.patience}")
        if on_epoch is not None:
            on_epoch(epoch, vm)  # may raise (e.g. optuna.TrialPruned) to abort
        if bad >= cfg.patience:
            log_fn(f"[{model_name}] early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device, cfg.amp)
    return model, {"best_val_pqrst_pearson": best_metric,
                   "test_pqrst_pearson": test.get("pqrst_pearson"),
                   "test_pearson": test.get("pearson"),
                   "test_mse": test.get("mse"), "test_rmse": test.get("rmse"),
                   "test_snr": test.get("snr"),
                   "test_r2": test.get("r2"), "test_prd": test.get("prd"),
                   "test_windows": test.get("n_windows"),
                   "test_pqrst_windows": test.get("n_pqrst_windows")}
