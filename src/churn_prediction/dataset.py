from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path


DEFAULT_DATA_URL = (
    "https://huggingface.co/datasets/arti199919/synthetic-saas-churn-sample/"
    "resolve/main/train.parquet?download=true"
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(dest_path: Path, url: str = DEFAULT_DATA_URL, overwrite: bool = False) -> dict:
    """
    Download the public SaaS churn dataset (Parquet) to `dest_path`.

    Returns basic metadata about the downloaded file (size + sha256).
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not overwrite:
        return {
            "path": str(dest_path),
            "bytes": dest_path.stat().st_size,
            "sha256": _sha256_file(dest_path),
            "downloaded": False,
            "url": url,
        }

    req = urllib.request.Request(url, headers={"User-Agent": "churn-prediction/0.1"})
    with urllib.request.urlopen(req) as response:  # noqa: S310 (trusted URL, user-overridable)
        with dest_path.open("wb") as f:
            shutil.copyfileobj(response, f)

    return {
        "path": str(dest_path),
        "bytes": dest_path.stat().st_size,
        "sha256": _sha256_file(dest_path),
        "downloaded": True,
        "url": url,
    }
