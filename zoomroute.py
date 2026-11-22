"""Zoom maps used DIRECTLY for rectangles (not as box seeds), with a router, CV on the real train.
For images the gate calls partial and that got no accepted box (pipecv logic, brute boxes):
  candidates: base = post-processed normal map (current) | zrect = best max-mean rectangle on the zoom map
  (rectangle maximising sum(zoom - c) over a coarse grid) | zthr = zoom map > t (largest component).
  router feature r = max(smoothed zoom map) - max(smoothed normal map) (rectangles light up at zoom, objects at zoom 1).
  Use the zoom candidate when r > rho; (candidate, rho, t, c) fit on the training folds.
    python zoomroute.py --method stegastamp --map ens2_stegastamp --zoom e2eR512s1Z_stegastamp"""
import argparse, itertools
import numpy as np
from scipy import ndimage as ndi
import data
from diag_full import cv_masks, score
from diagnostics import best_gate
from evalmaps import load_maps
from recteval import box_mask


def maxrect(z, c):
    """Rectangle maximising sum(z - c) (integral image, sides on a 4-cell grid)."""
    v = z - c; I = np.pad(v.cumsum(0).cumsum(1), ((1, 0), (1, 0))); best = (-1e9, None); r = range(0, 129, 4)
    for y0, y1 in itertools.combinations(r, 2):
        for x0, x1 in itertools.combinations(r, 2):
            s = I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]
            if s > best[0]:
                best = (s, (y0, x0, y1, x1))
    m = np.zeros((128, 128), bool)
    if best[1]:
        y0, x0, y1, x1 = best[1]; m[y0:y1, x0:x1] = True
    return m


ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--map", required=True); ap.add_argument("--zoom", required=True)
a = ap.parse_args(); df = data.train_table().set_index("name")
names, P, G, M, prm, full, gsv = cv_masks(a.map, a.method); fold = df.loc[names].fold.values
zn, Z = load_maps(a.zoom, "train"); Z = dict(zip(zn, Z)); Z = np.stack([Z[n] for n in names])
gf = np.zeros(len(P), bool)
for k in range(5):
    tr = fold != k; gf[~tr] = (gsv > best_gate(gsv[tr], full[tr]))[~tr]
part = ~gf; base = np.array([score(M[i], G[i]) for i in range(len(P))])
# boxes as in the current pipeline (fit tau per fold)
z = np.load(data.RESULTS / f"brute_{a.method}_train.npz"); bx = dict(zip(z["names"].tolist(), z["box"])); bv = dict(zip(z["names"].tolist(), z["score"]))
bs = np.array([bv.get(n, -np.inf) for n in names]); bsc = np.array([score(box_mask(bx[n]), G[i]) if n in bx else 0 for i, n in enumerate(names)])
r = np.array([ndi.gaussian_filter(Z[i], 1.5).max() - ndi.gaussian_filter(P[i], 1.5).max() for i in range(len(P))])
CS = (0.3, 0.4, 0.5, 0.6); TT = (0.3, 0.4, 0.5, 0.6)
cand = {}
for c in CS:
    cand[("rect", c)] = np.array([score(maxrect(Z[i], c), G[i]) if part[i] else 0 for i in range(len(P))])
for t in TT:
    def thr(i, t=t):
        m = Z[i] > t; lab, n = ndi.label(m)
        if n: m = lab == (1 + np.argmax(ndi.sum(m, lab, range(1, n + 1))))
        return m
    cand[("thr", t)] = np.array([score(thr(i), G[i]) if part[i] else 0 for i in range(len(P))])
RH = [np.inf] + list(np.quantile(r[part], np.linspace(0, 0.95, 20)))
TB = [np.inf] + list(np.unique(np.quantile(bs[np.isfinite(bs)], np.linspace(0, 1, 21))))
held = base.copy(); held_box = base.copy(); ch = []
for k in range(5):
    tr = fold != k; te = ~tr
    tb = max(TB, key=lambda t: np.where(part & (bs > t), bsc, base)[tr].mean())
    ub = part & (bs > tb); cur = np.where(ub, bsc, base); held_box[te] = cur[te]
    best = (cur[tr].mean(), None, np.inf)
    for key, cs in cand.items():
        for rho in RH:
            v = np.where(part & ~ub & (r > rho), cs, cur)[tr].mean()
            if v > best[0]:
                best = (v, key, rho)
    _, key, rho = best
    held[te] = (np.where(part & ~ub & (r > rho), cand[key], cur) if key else cur)[te]
    ch.append((key, round(float(rho), 3), int((part & ~ub & (r > rho))[te].sum()) if key else 0))
print(f"{a.method}: map+boxes CV {held_box.mean():.4f} -> + zoom route {held.mean():.4f} | partial {held_box[~full].mean():.4f} -> {held[~full].mean():.4f}")
print("   per fold (candidate, rho, #used on held-out):", ch)
