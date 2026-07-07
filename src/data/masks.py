"""PQRST-complex mask utilities (tensor-only, differentiable-friendly)."""
from __future__ import annotations

import torch


def build_pqrst_complex_mask(mask: torch.Tensor) -> torch.Tensor:
    """Contiguous P→T complex mask per beat.

    mask: (B, 4, L) with channels P, QRS, T, Other.
    returns: (B, L) bool mask, True from the start of each P to the end of its T.
    """
    if mask.dim() == 4:
        mask = mask.squeeze(2)
    p = mask[:, 0, :] > 0
    t = mask[:, 2, :] > 0
    B, L = p.shape
    zero_col = torch.zeros((B, 1), dtype=torch.bool, device=mask.device)

    prev_p = torch.cat([zero_col, p[:, :-1]], dim=1)
    p_start = p & (~prev_p)
    next_t = torch.cat([t[:, 1:], zero_col], dim=1)
    t_end = t & (~next_t)

    p_count = torch.cumsum(p_start.to(torch.int32), dim=1)
    t_end_before = torch.cumsum(t_end.to(torch.int32), dim=1) - t_end.to(torch.int32)
    complex_mask = p_count > t_end_before

    t_end_remaining = torch.flip(
        torch.cumsum(torch.flip(t_end.to(torch.int32), dims=[1]), dim=1), dims=[1]
    ) > 0
    return complex_mask & t_end_remaining
