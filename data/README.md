# Data Boundary

The full paired ear-ECG and chest-reference ECG dataset is private and is not
publicly released from this repository.

The private paired dataset used for the reported aggregate benchmark was
collected by [EDABK Research Lab](https://sites.google.com/set.hust.edu.vn/hust-edabk-lab/),
School of Electrical and Electronic Engineering, Hanoi University of Science and
Technology (HUST). This attribution identifies the collecting lab; it does not
make the raw subject-level recordings public.

Public data under this directory is limited to documentation, placeholders, and
synthetic or tiny anonymized examples that demonstrate code behavior. Public
sample or synthetic data must not be used to claim scientific performance.

## Public Folders

- `data/sample/`: optional tiny anonymized examples, if explicitly approved for
  release.
- `data/synthetic/`: generated ECG-like toy data for smoke/demo runs.
- `data/private/`: local-only placeholder for authorized private data. Its
  contents are ignored by git.

## Private Data Schema Used by Current Loaders

The current loader expects a private data root containing headerless two-column
CSV files named:

```text
ecg_data_01.csv
ecg_data_02.csv
...
```

Each file is sampled at 250 Hz and uses:

| Column | Signal | Role |
| --- | --- | --- |
| 0 | Ear ECG | model input `x` |
| 1 | Chest-reference ECG | target `y` |

Preprocessing caches are generated separately as `.npz` files containing
continuous filtered signals and masks:

| Array | Meaning | Shape |
| --- | --- | --- |
| `ear` | filtered and trimmed ear ECG | `(T,)` |
| `chest` | filtered and trimmed chest-reference ECG | `(T,)` |
| `masks` | P/QRS/T/Other masks | `(4, T)` |

## Reproducing Full Results Locally

To reproduce the full LOSO benchmark, place the authorized private dataset under
`data/private/` using the schema above, or pass a private path with
`--data-root`.

Example:

```bash
python scripts/run_loso.py --data-root /path/to/authorized/private_dataset
```

The public synthetic/demo data only verifies that loading, preprocessing,
windowing, model forward pass, and metric code run end-to-end. It is not a
scientific benchmark.
