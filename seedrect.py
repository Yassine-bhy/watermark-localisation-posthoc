"""AMELIORATIONS 2b (adapted after diag2a): the StegaStamp score peak is ~4 px wide (8 px shift -> 3 % of the score),
so a coarse grid jumps over it. Seed boxes from our segmentation map, then search densely around each seed:
  stage 1: shifts -32..32 step 8 (x and y) x scales {0.8, 0.9, 1, 1.1, 1.25} (h and w independently);
  stage 2: top 5 refined with edge / translation / scaling moves at 8, 4, 2, 1 px (bruterect.refine).
    python seedrect.py --method stegastamp --split train --maps ens2_stegastamp
Writes results/seedrect_<method>_<split>.npz (same format as bruterect: names, box, score, agree)."""
import argparse, time
import numpy as np, torch
from scipy import ndimage as ndi
import data
from bruterect import Scorer, ok, refine, iou
from evalmaps import load_maps

ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--maps", required=True)
a = ap.parse_args(); t0 = time.time()
sc = Scorer(a.method); lim = 2.2 if a.method == "trustmark" else 16.5
gs = dict(zip(*[np.load(data.RESULTS / "signals" / f"{a.method}_{a.split}.npz")[k].tolist() for k in ("names", "global_soft")]))
mn, mp = load_maps(a.maps, a.split); M = dict(zip(mn, mp))
ds = data.load_split(a.split, [a.method]); out = {"names": [], "box": [], "score": [], "agree": []}


def seeds(p):
    out = []
    for t in (0.3, 0.5, 0.7):
        lab, n = ndi.label(p > t)
        for j in range(1, n + 1):
            ys, xs = np.nonzero(lab == j)
            if len(ys) >= 4:
                out.append((ys.min() * 4, xs.min() * 4, (ys.max() + 1) * 4, (xs.max() + 1) * 4))
    out = sorted(set(out), key=lambda b: -p[b[0] // 4:b[2] // 4, b[1] // 4:b[3] // 4].mean())
    keep = []
    for b in out:
        if all(iou(b, k) < 0.7 for k in keep):
            keep.append(b)
    return keep[:4]


for i, m in enumerate(ds.meta):
    n = m["image_name"]
    if gs[n] > 0.6:
        continue
    it = ds[i]; x = torch.from_numpy(it["image"].copy()).permute(2, 0, 1)[None].float().div(255).cuda()
    s = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).cuda()
    cands = []
    for y0, x0, y1, x1 in seeds(M[n]):
        cy, cx, h, w = (y0 + y1) / 2, (x0 + x1) / 2, y1 - y0, x1 - x0
        for fh in (0.8, 0.9, 1.0, 1.1, 1.25):
            for fw in (0.8, 0.9, 1.0, 1.1, 1.25):
                for dy in range(-32, 33, 8):
                    for dx in range(-32, 33, 8):
                        b = (int(cy + dy - h * fh / 2), int(cx + dx - w * fw / 2), int(cy + dy + h * fh / 2), int(cx + dx + w * fw / 2))
                        if ok(b, lim):
                            cands.append(b)
    cands = list(set(cands))
    if not cands:
        best = ((0, 0, 512, 512), -1e9)
    else:
        v, _ = sc.score(x, cands, s); top = []
        for j in np.argsort(-v):
            if all(iou(cands[j], cands[t]) < 0.5 for t in top):
                top.append(j)
            if len(top) == 5:
                break
        best = max((refine(sc, x, s, cands[j], float(v[j]), lim) for j in top), key=lambda r: r[1])
        _, ag = sc.score(x, [best[0]], s)
        best = (best[0], best[1], float(ag[0]))
    for k, val in zip(("names", "box", "score", "agree"), (n, best[0], best[1], best[2] if len(best) > 2 else -1.0)):
        out[k].append(val)
    if len(out["names"]) % 10 == 0:
        print(len(out["names"]), f"{time.time() - t0:.0f}s", flush=True)
np.savez(data.RESULTS / f"seedrect_{a.method}_{a.split}.npz", **{k: np.array(v) for k, v in out.items()})
print("done", len(out["names"]), f"{time.time() - t0:.0f}s")
