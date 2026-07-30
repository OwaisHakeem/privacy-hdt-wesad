# Privacy-preserving federated learning on WESAD

Code and analysis scripts for the study *"Federation and privacy penalise
different patient subgroups through different mechanisms: a matched-configuration
study of differentially private federated learning on wearable physiological
data."*

This repository contains everything needed to reproduce the experiments and
figures. It does **not** contain the WESAD dataset (see
[Data](#data) below).

## What the study does

Using the WESAD dataset, the pipeline holds the model, the data partition and
the evaluation protocol fixed while varying one component at a time:

1. Centralised training (pooled-data baseline)
2. Federated averaging (FedAvg)
3. User-level (participant-level) differential privacy
4. Sample-level (window-level) differential privacy

and then mounts a membership-inference attack against both private mechanisms.
This isolates three effects that a single privacy–utility number conflates:
where federation harms, where privacy harms, and how far the formal privacy
budget overstates the empirically measured risk.

## Requirements

See `requirements.txt`. Core environment used for the reported results:

- Python 3.11
- PyTorch 2.11.0 (CUDA 12.8)
- Opacus 1.6.0
- NumPy, scikit-learn, matplotlib
- A single NVIDIA GeForce RTX 4080 SUPER GPU (CPU also works, more slowly)

Install:

```bash
pip install -r requirements.txt
```

## Data

The WESAD dataset is **not** redistributed here. Download it from the official
source and place it where the config expects:

- Schmidt et al., *Introducing WESAD, a Multimodal Dataset for Wearable Stress
  and Affect Detection*, ICMI 2018.
  DOI: [10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985)

Then build the windowed feature set with the dataset script (below).

## Pipeline

Scripts are numbered in run order. Configuration lives in `configs/`.

| Step | Script | Produces |
|------|--------|----------|
| Build dataset | `scripts/01_build_dataset.py` | windowed features |
| Create splits | `scripts/02_create_splits.py` | LOSO folds |
| Centralised baseline | `scripts/03_centralised_baseline.py` | Stage A |
| Local-only baseline | `scripts/04_local_only_baseline.py` | Stage A |
| FedAvg baseline | `scripts/05_fedavg_baseline.py` | Stage B |
| User-level DP sweep | `scripts/06_dp_fedavg_sweep.py` | Stage C |
| Membership inference | `scripts/07_membership_inference.py` | Stage D |
| Sample-level DP sweep | `scripts/08_dp_sample_sweep.py` | Stage C-prime |

Core modules are in `src/` (data handling, model, federated training, the two
DP mechanisms, metrics, and the attack).

Figures are generated from the stored per-run results by the scripts in
`paper_figures/`; each figure loads its values directly from the result JSON
files, so the figures trace to source data rather than to transcribed numbers.

## Reproducibility

Every condition runs under fixed random seeds. Reported results are 15
leave-one-subject-out folds × 5 seeds per condition. Differential-privacy
guarantees are verified against the accountant with a self-test before any
result is accepted.

## Citation

If you use this code, please cite the paper (details to be added on
publication).

## Licence

MIT — see `LICENSE`.
