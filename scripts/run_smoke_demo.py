"""Run a lightweight public smoke demo on synthetic/sample data.

This is not a benchmark. It performs preprocessing, windowing, a model forward
pass, basic metric computation, and writes small output artifacts.

Usage:
    python scripts/run_smoke_demo.py --data-root data/synthetic/demo --output-dir outputs/smoke_demo
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import PreprocessConfig, build_cache, load_subject, window_subject  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training.metrics import batch_metrics  # noqa: E402


def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _missing_data_message(data_root: Path) -> str:
    return (
        f"No ECG CSV files found under {data_root}.\n"
        "Generate synthetic demo data with:\n"
        "  python scripts/make_synthetic_data.py --output data/synthetic/demo"
    )


def _write_preview(path: Path, x, y, pred) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "ear_input", "chest_target", "model_output"])
        for idx, (xv, yv, pv) in enumerate(zip(x, y, pred)):
            writer.writerow([idx, f"{float(xv):.8f}", f"{float(yv):.8f}", f"{float(pv):.8f}"])


def _write_reconstruction_figure(path: Path, x, y, pred, fs: int, model_name: str) -> None:
    t = np.arange(len(x), dtype=np.float32) / fs
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, x, color="#4b5563", linewidth=1.2)
    axes[0].set_ylabel("Ear input")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, pred, color="#dc2626", linewidth=1.2)
    axes[1].set_ylabel("Model output")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, y, color="#1d4ed8", linewidth=1.2)
    axes[2].set_ylabel("Chest GT")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        f"Synthetic/sample smoke demo - {model_name.upper()} forward pass\n"
        "Demo only: not benchmark evidence and not clinical ECG",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/synthetic/demo")
    ap.add_argument("--output-dir", default="outputs/smoke_demo")
    ap.add_argument("--cache-dir", default="",
                    help="preprocessing cache directory; defaults to <output-dir>/cache")
    ap.add_argument("--model", default="sdcae")
    ap.add_argument("--max-windows", type=int, default=8)
    args = ap.parse_args()

    data_root = _resolve_project_path(args.data_root)
    output_dir = _resolve_project_path(args.output_dir)
    cache_dir = _resolve_project_path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreprocessConfig(train_overlap=0.5, test_overlap=0.5)
    subjects = build_cache(data_root, cache_dir, cfg, force=True)
    if not subjects:
        raise SystemExit(_missing_data_message(data_root))

    ear, chest, masks = load_subject(cache_dir, subjects[0])
    x_np, y_np, m_np = window_subject(ear, chest, masks, cfg, overlap=0.5)
    if x_np.shape[0] == 0:
        raise SystemExit("No windows were produced from the demo data.")

    n = min(args.max_windows, x_np.shape[0])
    x = torch.from_numpy(x_np[:n, None, :])
    y = torch.from_numpy(y_np[:n, None, :])
    m = torch.from_numpy(m_np[:n])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, 1).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device)).float().cpu()

    metrics = batch_metrics(pred, y.float(), m.float())
    artifact = {
        "demo_only": True,
        "not_a_benchmark": True,
        "data_root": str(data_root),
        "subject_file": subjects[0],
        "model": args.model,
        "windows_used": n,
        "window_samples": x.shape[-1],
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    _write_preview(
        output_dir / "preview_window.csv",
        x[0, 0].numpy(),
        y[0, 0].numpy(),
        pred[0, 0].numpy(),
    )
    _write_reconstruction_figure(
        output_dir / "reconstruction_demo.png",
        x[0, 0].numpy(),
        y[0, 0].numpy(),
        pred[0, 0].numpy(),
        fs=cfg.fs,
        model_name=args.model,
    )

    print(f"Smoke demo complete using {n} synthetic/sample window(s).")
    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'preview_window.csv'}")
    print(f"Wrote {output_dir / 'reconstruction_demo.png'}")
    print("This smoke demo is not a benchmark and must not be used for scientific claims.")


if __name__ == "__main__":
    main()
