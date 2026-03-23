from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn_prediction.features import DROP_COLS, FeatureSpec, infer_feature_spec


@dataclass(frozen=True)
class BuildResult:
    pipeline: Pipeline
    feature_spec: FeatureSpec


def build_pipeline(df: pd.DataFrame) -> BuildResult:
    feature_spec = infer_feature_spec(df)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_spec.numeric),
            ("cat", categorical_transformer, feature_spec.categorical),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=None,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    return BuildResult(pipeline=pipeline, feature_spec=feature_spec)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

