"""Plot fold-level Pearson/PQRST-Pearson curves from a LOSO summary CSV.

This script plots metrics only; it does not read raw ECG. By default the x-axis
uses anonymized Fold 1..N labels instead of subject IDs.

Usage:
    python scripts/plot_fold_pearson.py --summary-path results/metrics/loso_summary.csv --output outputs/figures/fold_pearson.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SUMMARY = ROOT / "results" / "metrics" / "loso_summary.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "figures" / "fold_pearson.png"
MODEL_LABELS = {
    "sdcae": "SDCAE",
    "deep_mf": "Deep-MF",
    "dcae": "DCAE",
    "deep_mf_mini": "Deep-MF-mini",
}


def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_latest_rows(summary_path: Path, metric: str) -> pd.DataFrame:
    if not summary_path.exists():
        raise SystemExit(f"Summary CSV not found: {summary_path}")
    df = pd.read_csv(summary_path)
    required = {"model", "test_subject", metric}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"{summary_path} is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=[metric])
    if "completed" in df.columns:
        df = df.sort_values("completed").drop_duplicates(
            subset=["model", "test_subject"], keep="last"
        )
    return df


def _ordered_subjects(df: pd.DataFrame) -> list[str]:
    return sorted(df["test_subject"].unique())


def plot_fold_pearson(summary_path: Path, output_path: Path, metric: str,
                      models: list[str] | None, show_subject_ids: bool) -> None:
    df = _load_latest_rows(summary_path, metric)
    if models:
        df = df[df["model"].isin(models)]
    if df.empty:
        raise SystemExit("No rows left to plot after filtering.")

    subjects = _ordered_subjects(df)
    x = np.arange(len(subjects))
    labels = subjects if show_subject_ids else [f"Fold {i + 1}" for i in range(len(subjects))]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for model in sorted(df["model"].unique()):
        model_df = df[df["model"] == model].set_index("test_subject")
        y = [model_df[metric].get(subject, np.nan) for subject in subjects]
        ax.plot(x, y, marker="o", linewidth=2, label=MODEL_LABELS.get(model, model))

    metric_label = "PQRST-Pearson" if metric == "test_pqrst_pearson" else "Full-window Pearson"
    ax.set_title(f"{metric_label} across LOSO folds")
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Held-out fold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0 if not show_subject_ids else 30, ha="center")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.text(
        0.01,
        0.01,
        "Aggregate/fold metric plot computed from private-dataset summary CSV. "
        "No raw ECG is shown.",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-path", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--metric", choices=("test_pqrst_pearson", "test_pearson"),
                    default="test_pqrst_pearson")
    ap.add_argument("--models", default="sdcae,deep_mf,dcae,deep_mf_mini",
                    help="comma-separated model keys, or empty for all models in the CSV")
    ap.add_argument("--show-subject-ids", action="store_true",
                    help="show private subject IDs on the x-axis; off by default")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    plot_fold_pearson(
        summary_path=_resolve_project_path(args.summary_path),
        output_path=_resolve_project_path(args.output),
        metric=args.metric,
        models=models or None,
        show_subject_ids=args.show_subject_ids,
    )
    print(f"Wrote {_resolve_project_path(args.output)}")


if __name__ == "__main__":
    main()
