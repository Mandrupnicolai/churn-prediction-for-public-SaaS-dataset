from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class Metrics:
    roc_auc: float
    pr_auc: float
    brier: float
    positive_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_binary_metrics(y_true, y_prob) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    return Metrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        positive_rate=float(y_true.mean()),
    )

