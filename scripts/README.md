# Script Guide

This folder contains command-line entry points for running experiments, demos,
plots, and result aggregation. Core reusable code lives in `src/`; scripts
should stay thin and call `src.data`, `src.models`, and `src.training`.

Keeping these files outside `src/` separates reusable library code from commands
with side effects such as reading private data, training models, and writing
outputs.

| Script | Purpose | Reads private data? | Typical output |
| --- | --- | --- | --- |
| `make_synthetic_data.py` | Generate small ECG-like synthetic CSV files for public demos. | No | `data/synthetic/demo/` |
| `run_smoke_demo.py` | Run preprocessing, windowing, one model forward pass, metrics, and a preview figure on synthetic/sample data. This is not benchmark evidence. | No by default | `outputs/smoke_demo/` |
| `run_loso.py` | Run leave-one-subject-out training and evaluation on the authorized paired ECG dataset. | Yes | `outputs/loso/metrics/loso_summary.csv` |
| `plot_reconstruction.py` | Train one LOSO fold and plot ear input, model output, and chest ground truth. | Yes | `outputs/figures/reconstruction.png` |
| `plot_fold_pearson.py` | Plot fold-level Pearson or PQRST-Pearson curves from a LOSO summary CSV. | No raw ECG; uses metrics CSV | `outputs/figures/fold_pearson.png` |
| `build_paper_artifacts.py` | Aggregate LOSO CSV results into paper/README-oriented tables and optional figures. | No raw ECG; uses metrics CSV | `outputs/paper_artifacts/` |
| `tune_loss_optuna.py` | Tune full combined-loss weights on one private validation fold with Optuna. | Yes | `outputs/optuna_loss_search/` |
| `tune_mse_pearson_optuna.py` | Tune the MSE-vs-Pearson loss trade-off on one private validation fold with Optuna. | Yes | `outputs/optuna_mse_pearson/` |

Do not commit generated output directories, private data, caches, Optuna study
databases, logs, or model checkpoints unless a release boundary is explicitly
documented.
