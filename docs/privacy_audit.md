# Privacy Audit

Date: 2026-07-08

Scope: local audit of `E:\SNN\CodeX` for raw ECG files, processed caches,
checkpoints, logs, notebooks, and result artifacts that may expose private or
subject-level information.

## Git Tracking Status

Git tracking status could not be checked from this workspace. Neither
`E:\SNN\CodeX` nor `E:\SNN` is a detectable Git working tree, so every
`tracked by git` entry below is marked `unknown`.

If this directory is later placed inside a Git repository, run:

```bash
git status --short
git ls-files data/raw data/private data/processed checkpoints outputs results
```

## Findings

| Path | Evidence | Tracked by git | Recommended action |
| --- | --- | --- | --- |
| `data/raw/` | Headerless paired ECG CSV files named `ecg_data_01.csv` through `ecg_data_12.csv`. These appear to be full subject-level recordings. | unknown | Keep private. Do not publish. If tracked, run `git rm --cached -r data/raw`. |
| `data/processed/` | `.npz` preprocessing caches derived from private recordings. | unknown | Keep private/reproducible only. If tracked, run `git rm --cached -r data/processed`. |
| `data/private/` | Reserved local-only private data folder. | unknown | Keep ignored except `data/private/.gitkeep`. If tracked data appears here, run `git rm --cached -r data/private` and then re-add only `.gitkeep`. |
| `checkpoints/1_lead_ECGFounder.pth` | Large model checkpoint. It may be licensed or provenance-sensitive and should not be required for MSE-only public demos. | unknown | Keep local unless release rights are documented. If tracked, run `git rm --cached -r checkpoints`. |
| `results/metrics/loso_summary*.csv` | Fold-level rows include held-out subject identifiers and private-dataset metrics. | unknown | Keep private or regenerate aggregate-only tables before publication. If tracked, run `git rm --cached results/metrics/loso_summary.csv results/metrics/loso_summary_mse_pearson.csv`. |
| `results/metrics/*all_runs*.csv`, `results/metrics/*smoke*.csv`, `results/metrics/*benchmark*.csv`, `results/metrics/*ablation*.csv` | Experiment summaries may contain fold-level or run-level private-dataset outputs. | unknown | Review before publishing; keep aggregate-only public tables. If unsafe and tracked, run `git rm --cached results/metrics/<file>`. |
| `results/metrics/*.log`, `results/figures/*.log`, `results/optuna_*/run.log` | Logs may expose local paths, subject IDs, or run details. | unknown | Keep ignored/private. If tracked, run `git rm --cached results/metrics/*.log results/figures/*.log results/optuna_*/*.log`. |
| `results/optuna_loss_search/` and `results/optuna_mse_pearson/` | Optuna trials and SQLite studies from private-data tuning runs. | unknown | Keep private or publish only aggregate narrative. If tracked, run `git rm --cached -r results/optuna_loss_search results/optuna_mse_pearson`. |
| `results/tables/*_by_fold.*` | Fold-level tables may expose subject-level metrics. | unknown | Keep private or replace with aggregate-only summaries. If tracked, run `git rm --cached results/tables/*_by_fold.*`. |
| `results/figures/reconstruction.png` | Example reconstruction figure derived from a held-out private subject. | unknown | Replace with synthetic/sample demo figure or keep private. If tracked, run `git rm --cached results/figures/reconstruction.png`. |
| `results/figures/loso/per_subject.png` | Per-subject LOSO plot. | unknown | Keep private or replace with aggregate figure. If tracked, run `git rm --cached results/figures/loso/per_subject.png`. |
| `docs/figures/reconstruction_private_result.png` | README visual copied from a private LOSO benchmark run. It shows no raw full recording but is still private-dataset-derived. | unknown | Publish only if example-window visual release is approved; otherwise replace with synthetic/sample figure and run `git rm --cached docs/figures/reconstruction_private_result.png`. |
| `docs/figures/fold_pearson.png` | README fold-level PQRST-Pearson plot with anonymized fold labels. | unknown | Prefer aggregate-only release; publish only if fold-level metric release is acceptable. |
| `env/` | Local virtual environment. | unknown | Do not publish. If tracked, run `git rm --cached -r env`. |
| `__pycache__/` folders | Local Python bytecode caches. | unknown | Removed outside `env/` during cleanup. Do not publish if recreated. |

## Safe Public Structure

The public repository should keep:

- `data/README.md`;
- `data/sample/README.md`;
- `data/synthetic/README.md`;
- `data/private/.gitkeep`;
- code, configs, and documentation;
- aggregate-only result tables where release is acceptable.

## Notes

No hardcoded API keys, passwords, tokens, or obvious credential strings were
found in the source/docs scan that excluded local environments, raw data,
processed caches, and checkpoints.

Do not delete raw data automatically. Untracking with `git rm --cached` removes
files from Git history going forward while leaving local copies on disk.
