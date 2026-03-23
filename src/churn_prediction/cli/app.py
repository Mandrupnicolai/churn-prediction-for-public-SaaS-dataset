from __future__ import annotations

import json
from pathlib import Path

import typer

from churn_prediction.dataset import DEFAULT_DATA_URL, download_dataset
from churn_prediction.paths import get_paths
from churn_prediction.train import predict, train_from_parquet

app = typer.Typer(add_completion=False)


@app.command("download-data")
def download_data(
    dest: Path | None = typer.Option(None, help="Destination parquet path."),
    url: str = typer.Option(DEFAULT_DATA_URL, help="Dataset URL to download."),
    overwrite: bool = typer.Option(False, help="Overwrite existing file."),
):
    paths = get_paths()
    dest_path = dest or (paths.data_raw / "train.parquet")
    meta = download_dataset(dest_path, url=url, overwrite=overwrite)
    typer.echo(json.dumps(meta, indent=2))


@app.command("train")
def train(
    parquet: Path | None = typer.Option(None, help="Input parquet path."),
    seed: int = typer.Option(42, help="Random seed for user-level split."),
):
    paths = get_paths()
    parquet_path = parquet or (paths.data_raw / "train.parquet")
    out = train_from_parquet(parquet_path, paths=paths, seed=seed)
    typer.echo(f"Saved model: {out.model_path}")
    typer.echo(f"Saved metrics: {out.metrics_path}")
    typer.echo(json.dumps(out.metrics.to_dict(), indent=2))


@app.command("evaluate")
def evaluate():
    paths = get_paths()
    metrics_path = paths.artifacts / "metrics.json"
    if not metrics_path.exists():
        raise typer.Exit(code=2)
    typer.echo(metrics_path.read_text(encoding="utf-8"))


@app.command("predict")
def predict_cmd(
    parquet: Path | None = typer.Option(None, help="Input parquet path."),
    model: Path | None = typer.Option(None, help="Path to trained model."),
    out_csv: Path | None = typer.Option(None, help="Output CSV path."),
):
    paths = get_paths()
    parquet_path = parquet or (paths.data_raw / "train.parquet")
    model_path = model or (paths.artifacts / "model.joblib")
    out_path = out_csv or (paths.artifacts / "predictions.csv")
    saved = predict(model_path, parquet_path, output_csv=out_path)
    typer.echo(f"Wrote: {saved}")

