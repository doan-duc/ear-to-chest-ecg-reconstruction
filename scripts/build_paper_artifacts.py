"""Aggregate LOSO results into publication-oriented tables.

The default output is aggregate-only. Per-subject figures are opt-in and should
remain private unless release is explicitly approved.

Usage:
    python scripts/build_paper_artifacts.py --summary-path results/metrics/loso_summary.csv
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
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402

DEFAULT_SUMMARY = ROOT / "results" / "metrics" / "loso_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "paper_artifacts"

MODELS = {
    "sdcae": ("SDCAE (ours)", 4),
    "deep_mf": ("Deep-MF", 32),
    "dcae": ("DCAE", 32),
    "deep_mf_mini": ("Deep-MF-mini", 32),
}
REF = "sdcae"
PRIVATE_RESULT_NOTE = (
    "Note: aggregate metrics computed on the private paired ear-ECG and "
    "chest-reference ECG dataset. Raw subject-level recordings are not "
    "publicly released.\n\n"
)


def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def count_params(name: str) -> int:
    return sum(p.numel() for p in build_model(name, 1).parameters())


def model_size_kb(name: str, bits: int) -> float:
    return count_params(name) * bits / 8 / 1024.0


def load_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise SystemExit(f"No results found at {summary_path} - run scripts/run_loso.py first.")
    df = pd.read_csv(summary_path)
    df = df.dropna(subset=["test_pqrst_pearson"])
    if "completed" in df.columns:
        df = df.sort_values("completed").drop_duplicates(
            subset=["model", "test_subject"], keep="last"
        )
    return df


def per_model_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in MODELS:
        group = df[df["model"] == key]
        if group.empty:
            continue
        display_name, bits = MODELS[key]
        pqrst = group["test_pqrst_pearson"].to_numpy()
        mse = (
            pd.to_numeric(group["test_mse"], errors="coerce")
            if "test_mse" in group
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "key": key,
                "Model": display_name,
                "n_folds": len(group),
                "PQRST-Pearson": (
                    f"{pqrst.mean():.3f} +/- {pqrst.std(ddof=1):.3f}"
                    if len(group) > 1
                    else f"{pqrst.mean():.3f}"
                ),
                "Best": f"{pqrst.max():.3f}",
                "Pearson": f"{group['test_pearson'].mean():.3f}",
                "MSE": f"{mse.mean():.3f}" if mse.notna().any() else "n/a",
                "RMSE": f"{group['test_rmse'].mean():.3f}",
                "Params": f"{count_params(key):,}",
                "Size (KB)": f"{model_size_kb(key, bits):.1f}",
                "_mean": pqrst.mean(),
            }
        )
    out = pd.DataFrame(rows).sort_values("_mean", ascending=False)
    sdcae_kb = model_size_kb(REF, MODELS[REF][1])
    out["vs SDCAE"] = [
        f"{model_size_kb(k, MODELS[k][1]) / sdcae_kb:.1f}x" for k in out["key"]
    ]
    return out


def pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    ref = df[df["model"] == REF].set_index("test_subject")["test_pqrst_pearson"]
    rows = []
    for key, (display_name, _) in MODELS.items():
        if key == REF or df[df["model"] == key].empty:
            continue
        other = df[df["model"] == key].set_index("test_subject")["test_pqrst_pearson"]
        common = ref.index.intersection(other.index)
        if len(common) < 2:
            continue
        a = ref.loc[common].to_numpy()
        b = other.loc[common].to_numpy()
        t_value, p_value = stats.ttest_rel(a, b)
        rows.append(
            {
                "Comparison": f"SDCAE vs {display_name}",
                "n": len(common),
                "mean diff": f"{(a - b).mean():+.4f}",
                "t": f"{t_value:.3f}",
                "p-value": f"{p_value:.4f}",
                "significant (p<0.05)": "yes" if p_value < 0.05 else "no",
            }
        )
    return pd.DataFrame(rows)


def _to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, stem: str, table_dir: Path, tex: bool = False) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    show = df.drop(columns=[c for c in ("key", "_mean") if c in df.columns])
    (table_dir / f"{stem}.csv").write_text(show.to_csv(index=False), encoding="utf-8")
    (table_dir / f"{stem}.md").write_text(
        PRIVATE_RESULT_NOTE + _to_markdown(show), encoding="utf-8"
    )
    if tex:
        (table_dir / f"{stem}.tex").write_text(show.to_latex(index=False), encoding="utf-8")


def fig_per_subject(df: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(df["test_subject"].unique())
    fig, ax = plt.subplots(figsize=(9, 4))
    for key in MODELS:
        group = df[df["model"] == key].set_index("test_subject")["test_pqrst_pearson"]
        if group.empty:
            continue
        ys = [group.get(subject, np.nan) for subject in subjects]
        style = {"marker": "o", "linewidth": 2} if key == REF else {"marker": ".", "alpha": 0.8}
        ax.plot(range(len(subjects)), ys, label=MODELS[key][0], **style)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([subject.replace("ecg_data_", "S") for subject in subjects], rotation=0)
    ax.set_ylabel("PQRST-Pearson")
    ax.set_xlabel("Held-out subject")
    ax.set_title("Per-subject LOSO reconstruction quality")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "per_subject.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-path", default=str(DEFAULT_SUMMARY),
                    help="private LOSO summary CSV to aggregate")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="directory for aggregate tables and optional figures")
    ap.add_argument("--include-per-subject-figure", action="store_true",
                    help="write a per-subject figure; keep private unless release is approved")
    args = ap.parse_args()

    summary_path = _resolve_project_path(args.summary_path)
    output_dir = _resolve_project_path(args.output_dir)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures" / "loso"

    df = load_summary(summary_path)
    print(
        f"Loaded {len(df)} rows across {df['model'].nunique()} model(s), "
        f"{df['test_subject'].nunique()} subject(s)."
    )

    table = per_model_table(df)
    write_table(table, "loso_by_model", table_dir, tex=True)
    print("\n=== Per-model summary ===")
    print(table.drop(columns=["key", "_mean"]).to_string(index=False))

    pairwise = pairwise_tests(df)
    if not pairwise.empty:
        write_table(pairwise, "loso_pairwise_stats", table_dir)
        print("\n=== Paired t-tests (vs SDCAE) ===")
        print(pairwise.to_string(index=False))

    if args.include_per_subject_figure:
        fig_per_subject(df, figure_dir)
        print(f"\nWrote aggregate tables to {table_dir} and per-subject figure to {figure_dir}")
    else:
        print(f"\nWrote aggregate tables to {table_dir}")


if __name__ == "__main__":
    main()
