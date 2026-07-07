"""Training losses for ECG reconstruction.

The combined objective is:

    total = lambda_ * MSE + beta_ * perceptual + gamma_ * TV + alpha_ * Pearson

``other_loss`` controls whether the auxiliary perceptual, total-variation and
Pearson terms are active. When it is disabled, those component losses are
returned as zero tensors so logging keeps a stable shape.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..models.net1d import Net1D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOUNDER_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "1_lead_ECGFounder.pth"


def z_score_normalize_pt(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Z-score normalize along the sample axis."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / (std + eps)


class ECGFounderPerceptualLoss(nn.Module):
    """MSE between frozen ECGFounder feature vectors for prediction and target."""

    def __init__(self, checkpoint_path, device):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "ECGFounder perceptual loss is enabled, but the checkpoint was "
                f"not found at {checkpoint_path}. Use other_loss=0 for MSE-only "
                "training, or pass --founder-checkpoint with a local trusted checkpoint."
            )
        self.founder = Net1D(
            in_channels=1,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=150,
            return_features=True,
        ).to(device)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            # Older PyTorch versions do not support weights_only. Checkpoints
            # must be local/trusted when this fallback is used.
            checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dense.")}
        self.founder.load_state_dict(state_dict, strict=False)

        self.founder.eval()
        for param in self.founder.parameters():
            param.requires_grad = False

        self.mse = nn.MSELoss()

    def format_for_founder(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 2-second, 250 Hz windows to the 500 Hz ECGFounder input format."""
        x_500hz = F.interpolate(x, size=1000, mode="linear", align_corners=False)
        x_padded = F.pad(x_500hz, pad=(0, 4000), mode="constant", value=0.0)
        mean = x_padded.mean(dim=2, keepdim=True)
        std = x_padded.std(dim=2, keepdim=True)
        return (x_padded - mean) / (std + 1e-8)

    def forward(self, pred_ecg: torch.Tensor, target_ecg: torch.Tensor) -> torch.Tensor:
        pred_formatted = self.format_for_founder(pred_ecg)
        target_formatted = self.format_for_founder(target_ecg)
        _, pred_features = self.founder(pred_formatted)
        _, target_features = self.founder(target_formatted)
        return self.mse(pred_features, target_features)


class PearsonCorrelationLoss(nn.Module):
    """Full-window Pearson loss: ``1 - mean(batch Pearson r)``."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean(dim=-1, keepdim=True)
        target_mean = target.mean(dim=-1, keepdim=True)
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        cov = (pred_centered * target_centered).sum(dim=-1)
        pred_std = torch.sqrt((pred_centered ** 2).sum(dim=-1) + self.eps)
        target_std = torch.sqrt((target_centered ** 2).sum(dim=-1) + self.eps)
        pearson_r = cov / (pred_std * target_std)
        return 1.0 - pearson_r.mean()


class TotalVarianceLoss(nn.Module):
    """Total-variation smoothness loss."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(x[:, :, 1:] - x[:, :, :-1])
        return torch.mean(torch.sum(diff, dim=2))


class MSEPearsonLoss(nn.Module):
    """Lightweight objective: lambda_*MSE + alpha_*(1 - Pearson).

    No perceptual / TV terms, so it needs no ECGFounder checkpoint and is fast.
    Returns (total, mse, pearson_loss) so it drops into the same training loop as
    FinalECGCombinedLoss (which reads element [0]).
    """

    def __init__(self, lambda_: float = 1.0, alpha_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
        self.alpha_ = alpha_
        self.mse = nn.MSELoss()
        self.pearson = PearsonCorrelationLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask=None):
        loss_mse = self.mse(pred, target)
        loss_pear = self.pearson(pred, target)
        total = self.lambda_ * loss_mse + self.alpha_ * loss_pear
        return total, loss_mse, loss_pear


class FinalECGCombinedLoss(nn.Module):
    """Combined reconstruction objective used by the training loop.

    Returns:
        total_loss, loss_recon, loss_percept, loss_tv, loss_pearson
    """

    def __init__(
        self,
        checkpoint_path=None,
        device=None,
        lambda_: float = 1.0,
        beta_: float = 10.0,
        gamma_: float = 5e-8,
        alpha_: float = 0.0,
        other_loss: int = 0,
    ):
        super().__init__()
        self.lambda_ = lambda_
        self.beta_ = beta_
        self.gamma_ = gamma_
        self.alpha_ = alpha_
        self.other_loss = other_loss
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_FOUNDER_CHECKPOINT)

        self.l_recon = nn.MSELoss()
        self.l_tv = TotalVarianceLoss() if self.other_loss else None
        self.l_percept = (
            ECGFounderPerceptualLoss(self.checkpoint_path, self.device)
            if self.other_loss
            else None
        )
        self.l_pearson = PearsonCorrelationLoss() if self.other_loss else None

    def forward(self, pred_ecg: torch.Tensor, target_ecg: torch.Tensor):
        loss_recon = self.l_recon(pred_ecg, target_ecg)

        if self.other_loss:
            if self.l_tv is None or self.l_percept is None or self.l_pearson is None:
                raise RuntimeError("Auxiliary loss modules were not initialized.")
            loss_tv = self.l_tv(pred_ecg)
            loss_percept = self.l_percept(
                z_score_normalize_pt(pred_ecg),
                z_score_normalize_pt(target_ecg),
            )
            loss_pearson = self.l_pearson(pred_ecg, target_ecg)
        else:
            loss_tv = pred_ecg.new_tensor(0.0)
            loss_percept = pred_ecg.new_tensor(0.0)
            loss_pearson = pred_ecg.new_tensor(0.0)

        total_loss = (
            (self.lambda_ * loss_recon)
            + (self.beta_ * loss_percept)
            + (self.gamma_ * loss_tv)
            + (self.alpha_ * loss_pearson)
        )
        return total_loss, loss_recon, loss_percept, loss_tv, loss_pearson
