# Churn prediction (public SaaS dataset)

Last updated: 2026-03-23 (Europe/Copenhagen)

This project trains a churn model on a public SaaS churn dataset (monthly customer snapshots) and provides a small CLI for download/train/evaluate/predict.

## Dataset

Default dataset source: Hugging Face dataset **`arti199919/synthetic-saas-churn-sample`** (84.8k rows in the viewer).

- Viewer: https://huggingface.co/datasets/arti199919/synthetic-saas-churn-sample/viewer
- Download (default in this repo): `train.parquet` from the dataset repo.

The model predicts `churned_next_month` using features like plan type, usage, tickets, payment failures, NPS, and seats.

## Setup

Prereq: Python 3.10+ installed and available on PATH (e.g. `python`, `python3`, or `py`).

From `C:\Users\Nicolai Mandrup\Documents\Playground\churn-prediction-public-saas-dataset`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
```

## Quickstart

1) Download data (requires internet):

```powershell
churn download-data
```

2) Train + evaluate:

```powershell
churn train
churn evaluate
```

Artifacts are written to `churn_artifacts/` (model pipeline + metrics JSON).

## Commands

```text
churn download-data   # downloads to data/raw/train.parquet
churn train           # trains model, saves churn_artifacts/model.joblib
churn evaluate        # prints metrics and writes churn_artifacts/metrics.json
churn predict         # writes churn_artifacts/predictions.csv
```

## Testing

```powershell
pytest
ruff check .
```

## Notes / assumptions

- Train/test split is done by `user_id` to reduce leakage across time for the same customer.
- This is a baseline (Logistic Regression with preprocessing). It’s intended as a starting point for feature engineering and better temporal validation.
