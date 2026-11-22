"""Phase 0: CV mIoU of the current best maps (+ boxes) split method x regime, and missing points per cell.
    ../.venv/bin/python research/tests/state_table.py     (CPU, read-only, ~2 min)"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, data, postproc
from diagnostics import best_gate, miou_batch, auc
from evalmaps import gate_scores, load_maps
from recteval import box_mask
TAGS = {"pixelseal": ("ens3_pixelseal", True), "trustmark": ("ens2_trustmark", True), "stegastamp": ("ens2_stegastamp", False)}
df = data.train_table().set_index("name"); g = data.train_masks128(); tot = 0
print("method      regime   n   CV(map+post)  CV(+boxes)  minority-AUC  lost points (of 1.0 total)")
for m, (tag, usebox) in TAGS.items():
    names, P = load_maps(tag, "train"); names = np.array(names); gs = gate_scores(m, "train")
    fold = df.loc[names].fold.values; full = df.loc[names].full_watermark.values
    G = np.stack([g[n] for n in names]); gsv = np.array([gs[n] for n in names]); S = postproc.table(P, G)
    z = np.load(data.RESULTS / f"brute_{m}_train.npz"); bx = dict(zip(z["names"].tolist(), z["box"])); bv = dict(zip(z["names"].tolist(), z["score"]))
    has = np.array([n in bx for n in names]); val = np.array([bv.get(n, -1e9) for n in names])
    B = np.array([miou_batch(box_mask(bx[n])[None], G[i][None])[0] if has[i] else 0 for i, n in enumerate(names)])
    hm = np.zeros(len(P)); hb = np.zeros(len(P))
    for k in range(5):
        tr = fold != k; te = ~tr; pf = gsv > best_gate(gsv[tr], full[tr]); mp = np.zeros(len(P))
        for reg, regp in ((full, pf), (~full, ~pf)):
            mm = S[..., tr & reg].mean(-1); s, o, t = np.unravel_index(np.argmax(mm), mm.shape); mp[regp] = S[s, o, t, regp]
        cand = np.unique(np.quantile(val[has], np.linspace(0, 1, 41))).tolist() + [np.inf]
        c = cand[int(np.argmax([np.where(~pf & has & (val > c), B, mp)[tr].mean() for c in cand]))]
        hm[te] = mp[te]; hb[te] = np.where(~pf & has & (val > c), B, mp)[te]
    A = np.array([auc(P[i].ravel(), G[i].ravel().astype(bool)) for i in range(len(G))])
    for nm, sel in (("full", full), ("partial", ~full)):
        cur = (hb if usebox else hm)[sel].mean(); lost = (1 - cur) * sel.sum() / 510; tot += lost
        print(f"{m:11s} {nm:8s} {sel.sum():3d}  {hm[sel].mean():.3f}         {hb[sel].mean():.3f}       {np.median(A[sel]):.3f}         {lost:.3f}")
print(f"overall CV of current best = {1 - tot:.3f}")
