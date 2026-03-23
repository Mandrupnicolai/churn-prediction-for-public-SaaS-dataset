from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "churn_artifacts"


def get_paths() -> ProjectPaths:
    # .../<repo>/src/churn_prediction/paths.py -> repo root is parents[2]
    root = Path(__file__).resolve().parents[2]
    return ProjectPaths(root=root)
