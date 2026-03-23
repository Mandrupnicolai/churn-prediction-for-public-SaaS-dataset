from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def load_monthly_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "user_id" not in df.columns:
        raise ValueError("Expected column 'user_id' in dataset")
    if "churned_next_month" not in df.columns:
        raise ValueError("Expected label column 'churned_next_month' in dataset")
    return df


def split_by_user_id(
    df: pd.DataFrame,
    *,
    label_col: str = "churned_next_month",
    user_col: str = "user_id",
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitData:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")

    users = df[user_col].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(users)

    n_test = max(1, int(round(len(users) * test_size)))
    test_users = set(users[:n_test].tolist())

    is_test = df[user_col].isin(test_users)
    train_df = df.loc[~is_test].copy()
    test_df = df.loc[is_test].copy()

    y_train = train_df[label_col].astype(int)
    y_test = test_df[label_col].astype(int)

    X_train = train_df.drop(columns=[label_col])
    X_test = test_df.drop(columns=[label_col])

    return SplitData(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

