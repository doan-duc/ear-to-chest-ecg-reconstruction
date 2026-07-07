"""Leave-one-subject-out training from private paired ECG CSVs.

The public repository does not ship the full dataset. For full reproduction,
place the authorized private dataset under data/private/ or pass --data-root.

Usage:
    python scripts/run_loso.py --data-root /path/to/authorized/private_dataset
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import PreprocessConfig, build_cache, loso_splits, make_loaders  # noqa: E402
from src.training import TrainConfig, train_fold, MSEPearsonLoss  # noqa: E402

DEFAULT_FOUNDER_CHECKPOINT = Path("checkpoints") / "1_lead_ECGFounder.pth"
FIELDS = [
    "model",
    "test_subject",
    "best_val_pqrst_pearson",
    "test_pqrst_pearson",
    "test_pearson",
    "test_mse",
    "test_rmse",
    "test_snr",
    "test_r2",
    "test_prd",
    "completed",
]
DEFAULTS = {
    "mode": "private",
    "data_root": None,
    "output_dir": "outputs/loso",
    "cache_dir": None,
    "models": "sdcae,dcae,deep_mf,deep_mf_mini",
    "train_overlap": 0.9,
    "val_overlap": 0.9,
    "test_overlap": 0.5,
    "epochs": 200,
    "patience": 25,
    "batch_size": 128,
    "lambda_": 1.0,
    "beta": 10.0,
    "gamma": 5e-8,
    "alpha": 0.0,
    "other_loss": 0,
    "loss_type": "combined",
    "founder_checkpoint": str(DEFAULT_FOUNDER_CHECKPOINT),
    "summary_path": None,
    "force_cache": False,
    "limit_subjects": 0,
}


def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _coerce_config_value(raw: str):
    value = raw.strip().strip("'\"")
    low = value.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"null", "none"}:
        return None
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _normalize_config_key(key: str) -> str:
    key = key.strip().replace("-", "_")
    return "lambda_" if key == "lambda" else key


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    cfg_path = _resolve_project_path(path)
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")
    if cfg_path.suffix.lower() == ".json":
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {_normalize_config_key(str(k)): v for k, v in data.items()}

    data = {}
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        data[_normalize_config_key(key)] = _coerce_config_value(value)
    return data


def _default_data_root(mode: str) -> Path:
    if mode == "synthetic":
        return ROOT / "data" / "synthetic" / "demo"
    if mode == "sample":
        return ROOT / "data" / "sample"
    return ROOT / "data" / "private"


def _build_parser(defaults: dict) -> argparse.ArgumentParser:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--mode", choices=("private", "sample", "synthetic"), default=defaults["mode"])
    ap.add_argument("--data-root", default=defaults["data_root"],
                    help="directory containing ecg_data_*.csv files")
    ap.add_argument("--output-dir", default=defaults["output_dir"],
                    help="directory for generated benchmark outputs")
    ap.add_argument("--cache-dir", default=defaults["cache_dir"],
                    help="preprocessing cache directory; defaults to <output-dir>/cache")
    ap.add_argument("--models", default=defaults["models"])
    ap.add_argument("--train-overlap", type=float, default=defaults["train_overlap"])
    ap.add_argument("--val-overlap", type=float, default=defaults["val_overlap"])
    ap.add_argument("--test-overlap", type=float, default=defaults["test_overlap"])
    ap.add_argument("--epochs", type=int, default=defaults["epochs"])
    ap.add_argument("--patience", type=int, default=defaults["patience"])
    ap.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    ap.add_argument("--lambda", dest="lambda_", type=float, default=defaults["lambda_"],
                    help="FinalECGCombinedLoss lambda_ weight for MSE")
    ap.add_argument("--beta", type=float, default=defaults["beta"],
                    help="FinalECGCombinedLoss beta_ weight for ECGFounder perceptual loss")
    ap.add_argument("--gamma", type=float, default=defaults["gamma"],
                    help="FinalECGCombinedLoss gamma_ weight for total-variance loss")
    ap.add_argument("--alpha", type=float, default=defaults["alpha"],
                    help="FinalECGCombinedLoss alpha_ weight for Pearson loss")
    ap.add_argument("--other-loss", type=int, choices=(0, 1), default=defaults["other_loss"],
                    help="0 disables perceptual/TV/Pearson terms; 1 enables them")
    ap.add_argument("--loss-type", choices=("combined", "mse_pearson"),
                    default=defaults["loss_type"],
                    help="combined = FinalECGCombinedLoss; mse_pearson avoids ECGFounder")
    ap.add_argument("--founder-checkpoint", default=defaults["founder_checkpoint"],
                    help="path to 1_lead_ECGFounder.pth, relative to project root by default")
    ap.add_argument("--summary-path", default=defaults["summary_path"],
                    help="optional CSV summary path; defaults to <output-dir>/metrics/loso_summary.csv")
    ap.add_argument("--force-cache", action="store_true", default=defaults["force_cache"])
    ap.add_argument("--limit-subjects", type=int, default=defaults["limit_subjects"],
                    help="debug: only run the first N held-out folds")
    return ap


def append_row(summary_path: Path, row: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_path.exists()
    if exists:
        with summary_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)
        if any(field not in existing_fields for field in FIELDS):
            with summary_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                for old_row in existing_rows:
                    w.writerow({k: old_row.get(k, "") for k in FIELDS})
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def _missing_private_data_message(data_root: Path) -> str:
    return (
        f"No private ECG CSV files found under {data_root}.\n"
        "Expected headerless two-column files named ecg_data_*.csv.\n"
        "If you want to reproduce the full LOSO benchmark, place the authorized "
        "private dataset under data/private/ or pass --data-root "
        "/path/to/private_dataset.\n"
        "For a public demo, run:\n"
        "  python scripts/make_synthetic_data.py --output data/synthetic/demo\n"
        "  python scripts/run_smoke_demo.py --data-root data/synthetic/demo "
        "--output-dir outputs/smoke_demo"
    )


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    defaults = {**DEFAULTS, **_load_config(pre_args.config)}
    args = _build_parser(defaults).parse_args()

    output_dir = _resolve_project_path(args.output_dir)
    data_root = _resolve_project_path(args.data_root) if args.data_root else _default_data_root(args.mode)
    cache_dir = _resolve_project_path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    summary_path = (
        _resolve_project_path(args.summary_path)
        if args.summary_path
        else output_dir / "metrics" / "loso_summary.csv"
    )

    pcfg = PreprocessConfig(
        train_overlap=args.train_overlap,
        val_overlap=args.val_overlap,
        test_overlap=args.test_overlap,
    )
    print(f"Data root: {data_root}", flush=True)
    print(f"Cache: {cache_dir}", flush=True)
    subjects = build_cache(data_root, cache_dir, pcfg, force=args.force_cache)
    if not subjects:
        raise SystemExit(_missing_private_data_message(data_root))
    if len(subjects) < 2:
        raise SystemExit("LOSO needs at least two subjects/files named ecg_data_*.csv.")
    print(f"Loaded {len(subjects)} subject file(s).", flush=True)

    founder_checkpoint = _resolve_project_path(args.founder_checkpoint)
    tcfg = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lambda_=args.lambda_,
        beta_=args.beta,
        gamma_=args.gamma,
        alpha_=args.alpha,
        other_loss=args.other_loss,
        founder_checkpoint=str(founder_checkpoint),
    )
    criterion = None
    if args.loss_type == "mse_pearson":
        criterion = MSEPearsonLoss(lambda_=args.lambda_, alpha_=args.alpha)
        print(
            f"Loss: MSEPearsonLoss lambda={args.lambda_} alpha={args.alpha} "
            "(no perceptual/TV)",
            flush=True,
        )
    else:
        print(
            "Loss: FinalECGCombinedLoss "
            f"lambda={args.lambda_} beta={args.beta} gamma={args.gamma} "
            f"alpha={args.alpha} other_loss={args.other_loss} "
            f"founder={founder_checkpoint if args.other_loss else 'not required'}",
            flush=True,
        )
    print(f"Summary: {summary_path}", flush=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = loso_splits(subjects)
    if args.limit_subjects:
        splits = splits[:args.limit_subjects]

    results = {m: [] for m in models}
    for model_name in models:
        for test_sid, train_sids in splits:
            train_loader, val_loader, test_loader, _ = make_loaders(
                cache_dir, train_sids, test_sid, pcfg, batch_size=tcfg.batch_size
            )
            _, res = train_fold(
                model_name, train_loader, val_loader, test_loader, tcfg, criterion=criterion
            )
            row = {
                "model": model_name,
                "test_subject": test_sid,
                "completed": datetime.now().astimezone().isoformat(timespec="seconds"),
                **res,
            }
            append_row(summary_path, row)
            results[model_name].append(res)
            print(
                f"[DONE] {model_name} test={test_sid} "
                f"pqrst={res['test_pqrst_pearson']:.4f} "
                f"pearson={res['test_pearson']:.4f} "
                f"mse={res['test_mse']:.6f}",
                flush=True,
            )

    print("\n==== LOSO summary (mean test metrics) ====")
    for model_name, rows in results.items():
        rows = [r for r in rows if r is not None]
        if rows:
            import statistics

            pqrst = [r["test_pqrst_pearson"] for r in rows]
            pearson = [r["test_pearson"] for r in rows]
            mse = [r["test_mse"] for r in rows]
            pqrst_std = statistics.pstdev(pqrst) if len(pqrst) > 1 else 0.0
            print(
                f"  {model_name}: pqrst={statistics.mean(pqrst):.4f}+/-{pqrst_std:.4f} "
                f"pearson={statistics.mean(pearson):.4f} "
                f"mse={statistics.mean(mse):.6f} n={len(rows)}"
            )


if __name__ == "__main__":
    main()
