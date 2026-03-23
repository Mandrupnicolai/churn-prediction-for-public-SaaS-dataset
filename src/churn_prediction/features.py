from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DROP_COLS = {
    "churned_next_month",
    "user_id",
    "month",
    # "churned" is a near-leakage/degenerate indicator (already churned this month).
    "churned",
}


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]


def infer_feature_spec(df: pd.DataFrame) -> FeatureSpec:
    cols = [c for c in df.columns if c not in DROP_COLS]
    working = df[cols].copy()

    categorical = [c for c in working.columns if working[c].dtype == "object"]
    numeric = [c for c in working.columns if c not in categorical]

    return FeatureSpec(numeric=sorted(numeric), categorical=sorted(categorical))

