"""Scoring: the official per-image mIoU from the starter kit, plus per-method / per-regime tables
and a submission writer that checks the format before saving."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data  # noqa: F401  (puts the kit on sys.path)
from wmloc.utils import miou_image, score  # noqa: F401  official functions, never re-implemented


def per_image_miou(preds: dict, gts: dict, names) -> np.ndarray:
    """Official mIoU for each name (preds/gts: name -> 128x128 0/1)."""
    return np.array([miou_image(preds[n], gts[n]) for n in names])


def breakdown(df: pd.DataFrame, col: str = "miou") -> pd.DataFrame:
    """Mean of `col` per method x regime, plus per-method and overall rows."""
    t = df.pivot_table(index="method", columns="regime", values=col, aggfunc="mean")
    t["all"] = df.groupby("method")[col].mean()
    t.loc["ALL"] = [df[df.regime == r][col].mean() for r in t.columns[:-1]] + [df[col].mean()]
    return t.round(4)


def write_submission(preds: dict, path) -> Path:
    """Write {"image_names", "masks"} for every test image, after checking the format."""
    names = data.test_names()
    assert set(preds) == set(names), "predictions must cover every test image exactly once"
    masks = np.stack([np.asarray(preds[n], np.uint8) for n in names])
    assert masks.shape == (len(names), 128, 128) and set(np.unique(masks)) <= {0, 1}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": names, "masks": torch.from_numpy(masks).bool()}, path)
    back = torch.load(path, weights_only=True)            # same loader as the server
    assert back["masks"].shape == masks.shape and len(set(back["image_names"])) == len(names)
    assert path.stat().st_size < 20e6, "file over 20 MB"
    return path
