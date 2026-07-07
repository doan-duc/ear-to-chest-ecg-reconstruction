# Data Access

The full paired ear-ECG and chest-reference ECG dataset is private and is not
distributed with this repository.

If you have authorized local access to the dataset, the code supports local
reproduction by passing the dataset location with `--data-root`.

## Current Loader Schema

Inspection of `src/data/dataset.py` and `src/data/preprocessing.py` shows that
the current loader expects a flat directory of headerless two-column CSV files:

```text
data/private/
  ecg_data_01.csv
  ecg_data_02.csv
  ...
```

Each CSV is interpreted as:

| Column | Signal | Role |
| --- | --- | --- |
| 0 | Ear ECG | model input |
| 1 | Chest-reference ECG | reconstruction target |

The scripts preprocess these files into an ignored cache containing `.npz`
files with `ear`, `chest`, and `masks` arrays.

## Normalized Future Schema

If the project later moves away from the current flat loader, a clearer private
schema would be:

```text
data/private/
  subject_01/
    ear.csv
    chest.csv
  subject_02/
    ear.csv
    chest.csv
```

TODO: Align loaders and documentation before using this normalized schema.

## Full Benchmark

Example:

```bash
python scripts/run_loso.py --mode private --data-root /path/to/authorized/private_dataset --output-dir outputs/loso
```

If private data is missing, the benchmark should fail with a clear message
instead of silently falling back to public demo data.

## Public Demo

Use synthetic data only to demonstrate the pipeline:

```bash
python scripts/make_synthetic_data.py --output data/synthetic/demo
python scripts/run_smoke_demo.py --data-root data/synthetic/demo --output-dir outputs/smoke_demo
```
