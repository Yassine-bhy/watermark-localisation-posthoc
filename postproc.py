"""Step 4a: post-processing of probe maps, chosen inside the CV (training folds only), separately per regime.
    python postproc.py step3_probe_pixelseal ...
Map transforms: Gaussian smoothing (sigma in cells of the 128 grid). After thresholding: none / fill holes /
remove small components / keep largest component / bounding box of largest component (rectangle = outpainting).
Per regime (gate on the whole-image decode as in evalmaps.py) the best (sigma, op, threshold) on the training folds is
applied to the held-out fold."""
import itertools
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import data
from diagnostics import best_gate, miou_batch
from evalmaps import gate_scores, load_maps

TS = list(np.linspace(0.04, 0.96, 24))
SIGMAS = (0, 1.5, 3, 6, 10, 16)
OPS = ("none", "fill", "small", "largest", "bbox")


def op(m, kind):
    if kind == "none" or not m.any():
        return m
    if kind == "fill":
        return _fill_both(m)
    lab, n = ndi.label(m); sz = ndi.sum(m, lab, range(1, n + 1))
    if kind == "small":
        return np.isin(lab, 1 + np.nonzero(sz >= 64)[0])
    k = 1 + int(np.argmax(sz)); cc = lab == k
    if kind == "largest":
        return cc
    ys, xs = np.nonzero(cc); b = np.zeros_like(m); b[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return b


def _fill_both(m):
    """Remove small holes and small islands (< 64 cells, i.e. 0.4 % of the image)."""
    lab, n = ndi.label(~m); sz = ndi.sum(~m, lab, range(1, n + 1)); m = m | np.isin(lab, 1 + np.nonzero(sz < 64)[0])
    lab, n = ndi.label(m); sz = ndi.sum(m, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.nonzero(sz >= 64)[0])


def table(P, G):
    """score[s, o, t, i]: mIoU of image i for sigma s, op o, threshold t."""
    S = np.zeros((len(SIGMAS), len(OPS), len(TS), len(P)), np.float32)
    for si, sg in enumerate(SIGMAS):
        Ps = np.stack([ndi.gaussian_filter(p, sg) if sg else p for p in P])
        for ti, t in enumerate(TS):
            B = Ps > t
            for oi, o in enumerate(OPS):
                S[si, oi, ti] = miou_batch(np.stack([op(b, o) for b in B]), G)
    return S


def evaluate(tag):
    method = [m for m in data.METHODS if m in tag][0]
    df = data.train_table().set_index("name"); g = data.train_masks128()
    names, P = load_maps(tag, "train"); gs = gate_scores(method, "train"); names = np.array(names)
    fold = df.loc[names].fold.values; full = df.loc[names].full_watermark.values
    G = np.stack([g[n] for n in names]); gsv = np.array([gs[n] for n in names])
    S = table(P, G); held = np.zeros(len(P)); choice = []
    for k in range(5):
        tr = fold != k; te = ~tr
        pf = gsv > best_gate(gsv[tr], full[tr])
        for reg_true, reg_pred in ((full, pf), (~full, ~pf)):
            sel_tr = tr & reg_true                                  # fit on the true regime of the training images
            m = S[..., sel_tr].mean(-1); s, o, t = np.unravel_index(np.argmax(m), m.shape)
            sel_te = te & reg_pred
            held[sel_te] = S[s, o, t, sel_te]
            choice.append((k, "full" if reg_true is full else "part", SIGMAS[s], OPS[o], round(TS[t], 2)))
    base = S[0, 0]                                                  # no post-processing, same CV
    return dict(tag=tag, cv=held.mean(), cv_full=held[full].mean(), cv_part=held[~full].mean(),
                oracle_global=S.reshape(-1, len(P)).mean(1).max()), choice


def fit_all(P, G, gsv, full):
    """Parameters on ALL training images (for the submission): gate + per regime (sigma, op, threshold)."""
    S = table(P, G); prm = {"gate": best_gate(gsv, full)}
    for reg, sel in (("full", full), ("part", ~full)):
        m = S[..., sel].mean(-1); s, o, t = np.unravel_index(np.argmax(m), m.shape)
        prm[reg] = (SIGMAS[s], OPS[o], TS[t])
    return prm


def apply_all(P, gsv, prm):
    out = []
    for p, gv in zip(P, gsv):
        sg, o, t = prm["full"] if gv > prm["gate"] else prm["part"]
        out.append(op((ndi.gaussian_filter(p, sg) if sg else p) > t, o))
    return np.stack(out)


if __name__ == "__main__":
    rows = []
    for t in sys.argv[1:]:
        r, ch = evaluate(t); rows.append(r); print(t, ch, flush=True)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
