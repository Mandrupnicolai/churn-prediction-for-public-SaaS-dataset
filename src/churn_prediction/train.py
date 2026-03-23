from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from churn_prediction.data import load_monthly_parquet, split_by_user_id
from churn_prediction.metrics import Metrics, compute_binary_metrics
from churn_prediction.modeling import build_pipeline, prepare_features
from churn_prediction.paths import ProjectPaths, get_paths


@dataclass(frozen=True)
class TrainOutputs:
    model_path: Path
    metrics_path: Path
    metrics: Metrics


def train_from_parquet(
    parquet_path: Path,
    *,
    paths: ProjectPaths | None = None,
    seed: int = 42,
) -> TrainOutputs:
    paths = paths or get_paths()
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    df = load_monthly_parquet(parquet_path)
    split = split_by_user_id(df, seed=seed)
    build = build_pipeline(split.X_train)

    X_train = prepare_features(split.X_train)
    X_test = prepare_features(split.X_test)

    build.pipeline.fit(X_train, split.y_train)
    y_prob = build.pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_binary_metrics(split.y_test, y_prob)

    model_path = paths.artifacts / "model.joblib"
    metrics_path = paths.artifacts / "metrics.json"

    joblib.dump(
        {
            "pipeline": build.pipeline,
            "feature_spec": build.feature_spec,
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2) + "\n", encoding="utf-8")

    return TrainOutputs(model_path=model_path, metrics_path=metrics_path, metrics=metrics)


def load_model(model_path: Path):
    obj = joblib.load(model_path)
    return obj["pipeline"]


def predict(
    model_path: Path,
    parquet_path: Path,
    *,
    output_csv: Path,
) -> Path:
    model = load_model(model_path)
    df = load_monthly_parquet(parquet_path)

    X = prepare_features(df)
    prob = model.predict_proba(X)[:, 1]

    out = pd.DataFrame(
        {
            "user_id": df["user_id"],
            "month": df.get("month"),
            "churn_probability_next_month": prob,
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return output_csv
