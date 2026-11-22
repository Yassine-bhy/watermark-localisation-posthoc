"""Kill test (CPU, ~1 min): exact max-sum rectangle (integral image, all rectangles on a 64x64 grid) on our probability
maps for PARTIAL images whose truth is an exact rectangle. Score of a box = sum over cells of (p - tau).
Compared with the current post-processing on the same images (in-sample best threshold, optimistic for the baseline).
Pass threshold (set in advance): >= +0.03 mIoU on rectangle images for at least one method."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, data, torch, torch.nn.functional as F
from diagnostics import miou_batch
from evalmaps import load_maps
TAGS = {"pixelseal": "ens3_pixelseal", "trustmark": "ens2_trustmark", "stegastamp": "ens2_stegastamp"}
df = data.train_table().set_index("name"); g = data.train_masks128()
def maxrect(p, tau, lim):
    q = torch.from_numpy(p - tau)[None, None]; q = F.avg_pool2d(q, 2)[0, 0] * 4; n = q.shape[0]
    c = F.pad(q.cumsum(0).cumsum(1), (1, 0, 1, 0)); best = (-1e9, None); idx = torch.arange(n + 1)
    for h in range(1, n + 1):
        A = c[h:, :] - c[:-h, :]                                   # (n-h+1) x (n+1)
        for w in range(max(1, int(np.ceil(h / lim))), min(n, int(h * lim)) + 1):
            S = A[:, w:] - A[:, :-w]; v, k = S.flatten().max(0)
            if v > best[0]:
                y, x = divmod(int(k), S.shape[1]); best = (float(v), (y, x, y + h, x + w))
    m = np.zeros((128, 128), bool)
    if best[1]: y0, x0, y1, x1 = [2 * t for t in best[1]]; m[y0:y1, x0:x1] = True
    return m
for meth, tag in TAGS.items():
    names, P = load_maps(tag, "train"); lim = 2.2 if meth == "trustmark" else 16.5
    sel = []
    for i, n in enumerate(names):
        if df.loc[n].full_watermark: continue
        m = g[n].astype(bool); ys, xs = np.nonzero(m)
        if m.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) > 0.98: sel.append(i)
    G = np.stack([g[names[i]] for i in sel]).astype(bool); Ps = P[sel]
    base = max(miou_batch(Ps > t, G).mean() for t in np.linspace(0.04, 0.96, 24))
    res = {tau: miou_batch(np.stack([maxrect(p, tau, lim) for p in Ps]), G).mean() for tau in (0.2, 0.35, 0.5)}
    print(f"{meth:11s} rect n={len(sel)}  threshold baseline {base:.3f} | max-sum rectangle {({k: round(float(v), 3) for k, v in res.items()})}", flush=True)
