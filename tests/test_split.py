from __future__ import annotations

import pandas as pd

from churn_prediction.data import split_by_user_id


def test_split_by_user_id_no_overlap():
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
            "plan_type": ["Starter"] * 8,
            "sessions": [1, 2, 3, 4, 5, 6, 7, 8],
            "churned_next_month": [0, 0, 0, 1, 0, 0, 1, 0],
        }
    )
    split = split_by_user_id(df, test_size=0.25, seed=1)

    train_users = set(split.X_train["user_id"].unique().tolist())
    test_users = set(split.X_test["user_id"].unique().tolist())
    assert train_users.isdisjoint(test_users)


def test_split_by_user_id_keeps_label_alignment():
    df = pd.DataFrame(
        {
            "user_id": ["A", "B", "C", "D"],
            "month": pd.to_datetime(["2025-01-01"] * 4),
            "plan_type": ["Starter", "Pro", "Starter", "Pro"],
            "sessions": [1, 2, 3, 4],
            "churned_next_month": [0, 1, 0, 1],
        }
    )
    split = split_by_user_id(df, test_size=0.5, seed=0)
    assert len(split.X_train) == len(split.y_train)
    assert len(split.X_test) == len(split.y_test)

