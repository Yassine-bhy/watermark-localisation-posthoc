"""Kill test (written before running): edge-aware refinement of the current maps with a guided filter (He et al.)
using the image (gray, 4x4-cell mean, 128 grid) as guide, before the usual CV post-processing (postproc.evaluate).
Motivation: PixelSeal residual is x JND (edges), TrustMark residual smooth from 256 -> map may leak across edges.
PASS if one fixed (r, eps) raises mean full-regime CV over the 3 methods by >= 0.010 with no method dropping > 0.003."""
import sys, numpy as np
sys.path.insert(0, "."); import data, postproc, evalmaps
from scipy import ndimage as ndi
TAGS = {"pixelseal": "ens3", "trustmark": "ens4", "stegastamp": "ens4"}
def box(x, r): return ndi.uniform_filter(x, 2 * r + 1, mode="reflect")
def guided(I, p, r, eps):
    mI, mp = box(I, r), box(p, r); cIp = box(I * p, r) - mI * mp; vI = box(I * I, r) - mI * mI
    a = cIp / (vI + eps); b = mp - a * mI
    return box(a, r) * I + box(b, r)
guides = {}
for sp in ("train",):
    ds = data.load_split(sp)
    for i in range(len(ds)):
        it = ds[i]; g = it["image"].astype(np.float32).mean(-1) / 255
        guides[it["image_name"]] = g.reshape(128, 4, 128, 4).mean((1, 3))
orig = evalmaps.load_maps
SET = [(0, 0)] + [(r, e) for r in (2, 4, 8) for e in (1e-3, 1e-2, 1e-1)]
for m, t in TAGS.items():
    tag = f"{t}_{m}"; out = []
    for r, e in SET:
        def lm(tg, split, r=r, e=e):
            names, P = orig(tg, split)
            if r == 0: return names, P
            return names, np.stack([np.clip(guided(guides[n], p.astype(np.float32), r, e), 0, 1) for n, p in zip(names, P)])
        postproc.load_maps = lm
        res, _ = postproc.evaluate(tag); out.append((r, e, round(res["cv_full"], 4), round(res["cv"], 4)))
        print(m, r, e, "cv_full", round(res["cv_full"], 4), "cv", round(res["cv"], 4), flush=True)
