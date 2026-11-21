"""Data access: wraps the starter kit's loader (wmloc.data) and adds the train table and CV folds.

The shared data lives one level up in ../release (read-only for us); the kit's code in ../wmloc.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))
from wmloc.data import LocalisationDataset  # noqa: E402

DATA = TASK_DIR / "release"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
METHODS = ("pixelseal", "trustmark", "stegastamp")


def load_split(split: str, methods=None) -> LocalisationDataset:
    """The kit's dataset: items with image (512x512x3 uint8), image_name, method, message, and on train
    also mask (512x512), mask128 (official 128x128 grid) and full_watermark."""
    return LocalisationDataset(DATA, split, methods)


def train_masks128() -> dict:
    """name -> official 128x128 ground-truth mask (uint8 0/1)."""
    z = np.load(DATA / "train" / "masks_128.npz")
    return dict(zip(z["names"].tolist(), z["masks"]))


def test_names() -> list:
    return [m["image_name"] for m in json.loads((DATA / "test" / "metadata.json").read_text())]


def cv_folds(df: pd.DataFrame, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per row, stratified by method x full_watermark (each fold has the same mix)."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(df), int)
    for _, idx in df.groupby(["method", "full_watermark"]).indices.items():
        idx = rng.permutation(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def train_table(k: int = 5) -> pd.DataFrame:
    """One row per train image: name, method, full_watermark, wm_frac (on the 128 grid), fold."""
    meta = json.loads((DATA / "train" / "metadata.json").read_text())
    m128 = train_masks128()
    df = pd.DataFrame({"name": [m["image_name"] for m in meta],
                       "method": [m["method"] for m in meta],
                       "full_watermark": [bool(m.get("full_watermark", False)) for m in meta]})
    df["regime"] = np.where(df.full_watermark, "full", "partial")
    df["wm_frac"] = [m128[n].mean() for n in df.name]
    df["fold"] = cv_folds(df, k)
    return df


def test_table() -> pd.DataFrame:
    meta = json.loads((DATA / "test" / "metadata.json").read_text())
    return pd.DataFrame({"name": [m["image_name"] for m in meta], "method": [m["method"] for m in meta]})
