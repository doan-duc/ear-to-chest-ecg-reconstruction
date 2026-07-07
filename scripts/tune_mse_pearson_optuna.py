"""Optuna search over the MSE-vs-Pearson trade-off (lambda_*MSE + alpha_*Pearson).

This is the lightweight follow-up to tune_loss_optuna.py. The earlier search over
the full FinalECGCombinedLoss found the ECGFounder perceptual and TV terms are
pushed to ~0 -- they do not help. So here we keep only the two useful terms (MSE
and Pearson) and tune their weights. No perceptual term => no ECGFounder => fast.

Scope: one model (sdcae), one fold (held-out subject; objective = VALIDATION
PQRST-Pearson, never test). Results go to outputs/optuna_mse_pearson/ by
default and never touch the LOSO results or the earlier perceptual search.

Usage:
    python scripts/tune_mse_pearson_optuna.py --data-root /path/to/authorized/private_dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import PreprocessConfig, build_cache, make_loaders  # noqa: E402
from src.training import TrainConfig, train_fold, MSEPearsonLoss  # noqa: E402

DEFAULT_DATA_ROOT = ROOT / "data" / "private"
DEFAULT_OUT_DIR = ROOT / "outputs" / "optuna_mse_pearson"
STUDY_NAME = "sdcae_mse_pearson"
MODEL = "sdcae"


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
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR),
                    help="directory for Optuna artifacts")
    ap.add_argument("--cache-dir", default="",
                    help="preprocessing cache directory; defaults to <output-dir>/cache")
    ap.add_argument("--n-trials", type=int, default=250)
    ap.add_argument("--subject", default="", help="held-out subject (default: auto)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--search-overlap", type=float, default=0.5,
                    help="train/validation-window overlap for the search fold")
    ap.add_argument("--timeout", type=int, default=0)
    args = ap.parse_args()

    data_root = _resolve_project_path(args.data_root)
    out_dir = _resolve_project_path(args.output_dir)
    cache_dir = _resolve_project_path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    pcfg = PreprocessConfig(
        train_overlap=args.search_overlap,
        val_overlap=args.search_overlap,
        test_overlap=0.5,
    )
    subjects = build_cache(data_root, cache_dir, pcfg)
    if not subjects:
        raise SystemExit(_missing_data_message(data_root))
    if len(subjects) < 2:
        raise SystemExit("Optuna tuning needs at least two ecg_data_*.csv subject files.")
    test_sid = args.subject or subjects[0]
    train_sids = [s for s in subjects if s != test_sid]
    train_loader, val_loader, test_loader, _ = make_loaders(
        cache_dir, train_sids, test_sid, pcfg, batch_size=128)
    print(f"Fold: held-out={test_sid} | objective = validation PQRST-Pearson "
          f"| model={MODEL} | loss = lambda*MSE + alpha*Pearson (no perceptual/TV)", flush=True)

    def objective(trial: optuna.Trial) -> float:
        lam = trial.suggest_float("lambda_", 1e-1, 1e1, log=True)
        alpha = trial.suggest_float("alpha_", 1e-2, 1e2, log=True)
        cfg = TrainConfig(epochs=args.epochs, patience=args.patience)
        crit = MSEPearsonLoss(lambda_=lam, alpha_=alpha)

        def on_epoch(epoch: int, val_pqrst: float) -> None:
            trial.report(val_pqrst, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _, res = train_fold(MODEL, train_loader, val_loader, test_loader, cfg,
                            log_every=1000, log_fn=lambda *_: None,
                            on_epoch=on_epoch, criterion=crit)
        return float(res["best_val_pqrst_pearson"])

    storage = f"sqlite:///{(out_dir / 'study.db').as_posix()}"
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=storage, load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=6))

    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, args.n_trials - done)
    print(f"Study has {done} finished trials; running {remaining} more "
          f"(target {args.n_trials}).", flush=True)

    def log_cb(st: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
        try:
            best = st.best_value
        except ValueError:
            best = float("nan")
        print(f"[trial {tr.number}] state={tr.state.name} value={tr.value} "
              f"best={best:.5f}", flush=True)
        if st.best_trial is not None:
            (out_dir / "best_params.json").write_text(json.dumps(
                {"best_val_pqrst_pearson": st.best_value, **st.best_params},
                indent=2), encoding="utf-8")

    study.optimize(objective, n_trials=remaining,
                   timeout=(args.timeout or None), callbacks=[log_cb],
                   gc_after_trial=True)

    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    best = {"study": STUDY_NAME, "model": MODEL, "held_out_subject": test_sid,
            "objective": "validation PQRST-Pearson",
            "loss": "lambda*MSE + alpha*Pearson (no perceptual/TV)",
            "n_trials": len(study.trials),
            "best_val_pqrst_pearson": study.best_value, **study.best_params}
    (out_dir / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    try:
        imp = optuna.importance.get_param_importances(study)
        (out_dir / "param_importances.json").write_text(
            json.dumps({k: float(v) for k, v in imp.items()}, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"(param importance skipped: {e})", flush=True)

    print("\n==== BEST ====", flush=True)
    print(json.dumps(best, indent=2), flush=True)
    print(f"\nArtifacts in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
