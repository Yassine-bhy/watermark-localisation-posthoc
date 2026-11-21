"""Score any per-image probability maps with the same protocol, and turn the best ones into a submission.
    python evalmaps.py probe_pixelseal probe_deep_pixelseal ...     (tags = results/preds/<tag>_train.npz)
Protocol (5-fold CV, held-out only): full/partial gate from the whole-image decode (threshold fit on the training
folds), then one threshold per regime on the map (fit on the training folds). Prints within-image AUC and CV mIoU."""
import os
import sys

import numpy as np
import pandas as pd

import data
from diagnostics import auc, best_gate, miou_batch

TS = [-1] + list(np.linspace(0.02, 0.98, 49)) + [2]


GATE_MAP = dict(kv.split("=") for kv in os.environ.get("GATE_MAP", "").split(",") if kv)   # method=maptag


def gate_scores(method, split):
    """Full/partial gate score: whole-image decode agreement, or (GATE_MAP env, e.g. 'trustmark=ens2_trustmark')
    the mean of a probability map (more accurate for TM / SS: 0.988 / 0.994 vs 0.971 / 0.976 held-out)."""
    if method in GATE_MAP:
        z = np.load(data.RESULTS / "preds" / f"{GATE_MAP[method]}_{split}.npz")
        return dict(zip(z["names"].tolist(), z["prob"].astype(np.float32).mean((1, 2))))
    z = np.load(data.RESULTS / "signals" / f"{method}_{split}.npz")
    return dict(zip(z["names"].tolist(), z["global_soft"]))


def load_maps(tag, split):
    z = np.load(data.RESULTS / "preds" / f"{tag}_{split}.npz")
    return z["names"].tolist(), z["prob"].astype(np.float32)


def fit_params(P, G, gsv, full):
    prm = {"gate": best_gate(gsv, full)}
    for reg, sel in (("t_full", full), ("t_part", ~full)):
        prm[reg] = TS[int(np.argmax([miou_batch(P[sel] > t, G[sel]).mean() for t in TS]))]
    return prm


def apply(P, gsv, prm):
    pf = gsv > prm["gate"]
    return np.where(pf[:, None, None], P > prm["t_full"], P > prm["t_part"])


def evaluate(tag):
    method = [m for m in data.METHODS if m in tag][0]
    df = data.train_table().set_index("name"); g = data.train_masks128()
    names, P = load_maps(tag, "train"); gs = gate_scores(method, "train")
    names = np.array(names); fold = df.loc[names].fold.values; full = df.loc[names].full_watermark.values
    G = np.stack([g[n] for n in names]); gsv = np.array([gs[n] for n in names]); Pr = np.zeros_like(G, bool)
    for k in range(5):
        tr = fold != k
        Pr[~tr] = apply(P[~tr], gsv[~tr], fit_params(P[tr], G[tr], gsv[tr], full[tr]))
    sc = miou_batch(Pr, G)
    a = np.array([auc(P[i].ravel(), G[i].ravel().astype(bool)) for i in range(len(G))])
    return dict(tag=tag, method=method, auc_full=np.median(a[full]), auc_partial=np.median(a[~full]),
                cv=sc.mean(), cv_full=sc[full].mean(), cv_partial=sc[~full].mean()), Pr, names


if __name__ == "__main__":
    rows = [evaluate(t)[0] for t in sys.argv[1:]]
    print(pd.DataFrame(rows).round(4).to_string(index=False))
