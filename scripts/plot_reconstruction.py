"""Plot one reconstruction example: ear input, model output, chest ground truth.

Trains a single LOSO fold (default: sdcae, MSE loss) on a held-out subject, then
plots the best-reconstructed test window as three signals — the ear ECG that goes
in, the model's reconstruction, and the chest ECG ground truth.

Usage:
    python scripts/plot_reconstruction.py --data-root /path/to/authorized/private_dataset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import PreprocessConfig, build_cache, make_loaders  # noqa: E402
from src.training import TrainConfig, train_fold, pqrst_pearson  # noqa: E402

DEFAULT_DATA_ROOT = ROOT / "data" / "private"
DEFAULT_OUT = ROOT / "outputs" / "figures" / "reconstruction.png"
FS = 250  # Hz


def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _missing_data_message(data_root: Path) -> str:
    return (
        f"No private ECG CSV files found under {data_root}.\n"
        "Expected headerless two-column files named ecg_data_*.csv. "
        "Place the authorized private dataset under data/private/ or pass "
        "--data-root /path/to/private_dataset."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                    help="directory containing private ecg_data_*.csv files")
    ap.add_argument("--cache-dir", default="",
                    help="preprocessing cache directory; defaults to outputs/plot_cache")
    ap.add_argument("--subject", default="", help="held-out subject id (default: auto)")
    ap.add_argument("--model", default="sdcae")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--other-loss", type=int, choices=(0, 1), default=0)
    ap.add_argument("--founder-checkpoint",
                    default=str(Path("checkpoints") / "1_lead_ECGFounder.pth"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    data_root = _resolve_project_path(args.data_root)
    cache_dir = _resolve_project_path(args.cache_dir) if args.cache_dir else ROOT / "outputs" / "plot_cache"
    pcfg = PreprocessConfig()
    subjects = build_cache(data_root, cache_dir, pcfg)
    if not subjects:
        raise SystemExit(_missing_data_message(data_root))
    if len(subjects) < 2:
        raise SystemExit("Plotting a LOSO reconstruction needs at least two ecg_data_*.csv files.")
    test_sid = args.subject or subjects[0]
    train_sids = [s for s in subjects if s != test_sid]
    print(f"Held-out subject: {test_sid} | model: {args.model}", flush=True)

    train_loader, val_loader, test_loader, _ = make_loaders(
        cache_dir, train_sids, test_sid, pcfg, batch_size=128)

    fck = _resolve_project_path(args.founder_checkpoint)
    tcfg = TrainConfig(epochs=args.epochs, patience=args.patience,
                       other_loss=args.other_loss, founder_checkpoint=str(fck))
    model, res = train_fold(args.model, train_loader, val_loader, test_loader, tcfg)
    print(f"test PQRST-Pearson = {res['test_pqrst_pearson']:.4f}", flush=True)

    # find the single test window with the best PQRST-Pearson (cleanest example)
    device = next(model.parameters()).device
    model.eval()
    best = {"r": -2.0}
    with torch.no_grad():
        for x, y, m in test_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x).float()
            # score every window individually, keep the best one for a clean plot
            for i in range(x.shape[0]):
                ri = float(pqrst_pearson(pred[i:i+1], y[i:i+1].float(), m[i:i+1].float()))
                if ri > best["r"]:
                    best = {"r": ri,
                            "x": x[i, 0].cpu().numpy(),
                            "y": y[i, 0].cpu().numpy(),
                            "pred": pred[i, 0].cpu().numpy()}

    t = np.arange(best["x"].shape[0]) / FS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax1.plot(t, best["x"], color="#7f8c8d", linewidth=1.2)
    ax1.set_ylabel("Ear ECG\n(input)")
    ax1.set_title(f"Ear -> heart ECG reconstruction  |  {test_sid}  "
                  f"(PQRST-Pearson = {best['r']:.3f})")
    ax1.grid(alpha=0.3)

    ax2.plot(t, best["y"], color="#2c3e50", linewidth=1.6, label="Chest ECG (ground truth)")
    ax2.plot(t, best["pred"], color="#c0392b", linewidth=1.4, linestyle="--",
             label=f"{args.model.upper()} output")
    ax2.set_ylabel("Heart ECG")
    ax2.set_xlabel("Time (s)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = _resolve_project_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
