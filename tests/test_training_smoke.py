from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn_prediction.train import train_from_parquet


def test_train_from_parquet_smoke(tmp_path: Path):
    df = pd.DataFrame(
        {
            "user_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "month": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-02-01",
                    "2025-01-01",
                    "2025-02-01",
                    "2025-01-01",
                    "2025-02-01",
                    "2025-01-01",
                    "2025-02-01",
                ]
            ),
            "tenure_month": [1, 2, 1, 2, 1, 2, 1, 2],
            "plan_type": ["Starter", "Starter", "Pro", "Pro", "Starter", "Pro", "Pro", "Starter"],
            "monthly_price": [10.0, 10.0, 50.0, 50.0, 12.0, 55.0, 60.0, 11.0],
            "mrr": [10.0, 10.0, 50.0, 50.0, 12.0, 55.0, 60.0, 11.0],
            "sessions": [1, 2, 10, 11, 3, 4, 12, 13],
            "feature_usage_score": [0.1, 0.2, 0.9, 0.95, 0.3, 0.4, 0.8, 0.7],
            "support_tickets": [0, 0, 1, 1, 0, 0, 2, 2],
            "payment_failures": [0, 0, 0, 1, 0, 0, 1, 0],
            "nps_score": [7, 7, 2, 3, 6, 6, 1, 2],
            "product_incident": [0, 0, 0, 1, 0, 0, 1, 0],
            "active_seats": [1, 1, 10, 10, 2, 2, 12, 12],
            "churned": [0, 0, 0, 0, 0, 0, 0, 0],
            "churned_next_month": [0, 0, 0, 1, 0, 0, 1, 0],
        }
    )
    parquet_path = tmp_path / "train.parquet"
    df.to_parquet(parquet_path, index=False)

    project_root = tmp_path / "proj"
    project_root.mkdir()
    (project_root / "churn_artifacts").mkdir()
    out = train_from_parquet(
        parquet_path,
        paths=type("P", (), {"artifacts": project_root / "churn_artifacts"})(),  # minimal paths
        seed=0,
    )

    assert out.model_path.exists()
    assert out.metrics_path.exists()

