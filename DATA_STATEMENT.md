# Data Statement

## Dataset

Dataset name: private paired ear-ECG and chest-reference ECG dataset.

Provenance: the private paired dataset used for the reported aggregate
benchmark was collected by [EDABK Research Lab](https://sites.google.com/set.hust.edu.vn/hust-edabk-lab/),
School of Electrical and Electronic Engineering, Hanoi University of Science and
Technology (HUST).

Release status: the full raw dataset is not publicly released.

Reason: participant privacy and ECG data-governance constraints.

## Public Repository Contents

The public repository may contain:

- code and configuration files;
- documentation;
- aggregate benchmark metrics and tables;
- smoke/demo scripts;
- synthetic or tiny anonymized sample data for demonstrating pipeline behavior.

Synthetic/sample data is not scientifically meaningful and must not be used to
claim reconstruction performance.

## Private Contents

The private dataset contains raw subject-level paired recordings, with ear ECG
as input and simultaneous chest-reference ECG as target. These recordings must
remain outside public release unless a future approved data-sharing process
explicitly permits otherwise.

## Intended Use

This repository is intended for an offline research benchmark of
ear-to-chest-reference ECG morphology reconstruction.

## Not Intended For

This repository is not intended for:

- diagnosis;
- clinical decision-making;
- medical-device validation;
- arrhythmia detection;
- real-time deployment claims.

## Consent And Ethics

TODO: Document the exact consent, ethics, IRB, or institutional review details
that govern collection and use of the private dataset.

TODO: Document whether any anonymized sample data can be publicly released.

Until those details are documented, the full raw dataset and subject-level
recordings must remain private.

## Data Access

External access to the full private dataset is not provided through this
repository. Future access would require an approved data-sharing process,
including participant privacy review and any required institutional approvals.

## Reproducibility Scope

1. Public code-level and pipeline-level demonstration is supported using
   synthetic or approved sample data.
2. Full private-result reproduction requires authorized local access to the
   private dataset.
3. Public LOSO benchmark reporting is aggregate-only and should state that the
   metrics were computed on the private dataset.
