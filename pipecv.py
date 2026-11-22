"""Full pipeline CV (held-out, decisions fit on training folds), per method:
gate -> full: post-processed map | partial: box if its score > tau_box, else SAM pick if agree(pick, map blob) >= tau_sam,
else post-processed map.   SAM pick = research/tests/test_samprompt.py ('contrast' rule) from results/samcache_*.
    python pipecv.py --method trustmark --map ens2_trustmark [--box brute] [--sam 1]
Reports CV overall / full / partial and the chosen thresholds per fold."""
import argparse
import numpy as np
from scipy import ndimage as ndi
import data
from diag_full import cv_masks, score
from diagnostics import best_gate
from recteval import box_mask
from scipy import ndimage as ndi  # noqa
from evalmaps import load_maps


def sam_pick(cands, p, base, pt):
    q = ndi.gaussian_filter(p, 1.5); cands = [c for c in cands if c.any() and not c.all()]
    if not cands:
        return None, 0.0

    def contrast(c):
        ring = ndi.binary_dilation(c, iterations=6) & ~c
        return (q[c].mean() - q[ring].mean()) if ring.any() else -9
    pick = max(cands, key=contrast); lab, _ = ndi.label(base); y, x = pt
    comp = (lab == lab[y, x]) if lab[y, x] else np.zeros_like(base)
    return pick, (pick & comp).sum() / max((pick | comp).sum(), 1)


def maxrect(z, c):
    """Rectangle maximising sum(z - c) over a 4-cell grid (zoomroute.py)."""
    import itertools
    v = z - c; I = np.pad(v.cumsum(0).cumsum(1), ((1, 0), (1, 0))); best = (-1e9, None); r = range(0, 129, 4)
    for y0, y1 in itertools.combinations(r, 2):
        for x0, x1 in itertools.combinations(r, 2):
            s_ = I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]
            if s_ > best[0]:
                best = (s_, (y0, x0, y1, x1))
    m = np.zeros((128, 128), bool)
    if best[1]:
        y0, x0, y1, x1 = best[1]; m[y0:y1, x0:x1] = True
    return m


def zoom_feats(Z, P):
    """Router feature: max of the smoothed zoom map minus max of the smoothed normal map."""
    return np.array([ndi.gaussian_filter(z, 1.5).max() - ndi.gaussian_filter(p, 1.5).max() for z, p in zip(Z, P)])


def run(method, tag, box, use_sam, zoom="", split_report=True):
    df = data.train_table().set_index("name")
    names, P, G, M, prm, full, gsv = cv_masks(tag, method); fold = df.loc[names].fold.values
    gated_full = np.zeros(len(P), bool)
    for k in range(5):
        tr = fold != k; gated_full[~tr] = (gsv > best_gate(gsv[tr], full[tr]))[~tr]
    base = np.array([score(M[i], G[i]) for i in range(len(P))])
    bs = np.full(len(P), -np.inf); bsc = np.zeros(len(P))
    if box:
        z = np.load(data.RESULTS / f"{box}_{method}_train.npz"); bx = dict(zip(z["names"].tolist(), z["box"])); bv = dict(zip(z["names"].tolist(), z["score"]))
        for i, n in enumerate(names):
            if n in bx:
                bs[i] = bv[n]; bsc[i] = score(box_mask(bx[n]), G[i])
    ss = np.zeros(len(P)); sag = np.full(len(P), -1.0)
    if use_sam:
        z = np.load(data.RESULTS / f"samcache_{method}_train.npz"); cm = dict(zip(z["names"].tolist(), z["masks"])); cp = dict(zip(z["names"].tolist(), z["point"]))
        for i, n in enumerate(names):
            if n in cm:
                pick, ag = sam_pick(list(cm[n]), P[i], M[i], tuple(cp[n]))
                if pick is not None:
                    ss[i] = score(pick, G[i]); sag[i] = ag
    zr = np.full(len(P), -np.inf); zsc = np.zeros(len(P))
    if zoom:
        zn, Zm = load_maps(zoom, "train"); Zd = dict(zip(zn, Zm)); Z = np.stack([Zd[n] for n in names])
        zr = zoom_feats(Z, P); zsc = np.array([score(maxrect(Z[i], 0.3), G[i]) for i in range(len(P))])
    held = base.copy(); ch = []
    TB = [np.inf] + (list(np.unique(np.quantile(bs[np.isfinite(bs)], np.linspace(0, 1, 41)))) if box else [])
    TA = [np.inf] + ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] if use_sam else [])
    TZ = [np.inf] + (list(np.quantile(zr[~gated_full], np.linspace(0, 0.95, 20))) if zoom else [])
    for k in range(5):
        tr = fold != k; te = ~tr; part = ~gated_full
        best = None
        for tb in TB:
            ub = part & (bs > tb)
            for tz in TZ:
                uz = part & ~ub & (zr > tz)
                for ta in TA:
                    us = part & ~ub & ~uz & (sag >= ta)
                    v = np.where(ub, bsc, np.where(uz, zsc, np.where(us, ss, base)))[tr].mean()
                    if best is None or v > best[0]:
                        best = (v, tb, tz, ta)
        _, tb, tz, ta = best; ub = part & (bs > tb); uz = part & ~ub & (zr > tz); us = part & ~ub & ~uz & (sag >= ta)
        held[te] = np.where(ub, bsc, np.where(uz, zsc, np.where(us, ss, base)))[te]
        ch.append((round(float(tb), 2), round(float(tz), 3), ta, int(ub[te].sum()), int(uz[te].sum()), int(us[te].sum())))
    print(f"{method} map={tag} box={box} zoom={zoom} sam={use_sam}: CV {held.mean():.4f} (map only {base.mean():.4f}) | full {held[full].mean():.4f} | partial {held[~full].mean():.4f} (map only {base[~full].mean():.4f})")
    print("   per fold (tau_box, rho_zoom, tau_sam, #box, #zoom, #sam on held-out):", ch)
    return held.mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--map", required=True)
    ap.add_argument("--box", default=""); ap.add_argument("--sam", type=int, default=0); ap.add_argument("--zoom", default=""); a = ap.parse_args()
    run(a.method, a.map, a.box, bool(a.sam), a.zoom)
