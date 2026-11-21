"""AMELIORATIONS 2a: for every real train rectangle (partial image whose mask fills its bounding box), score the TRUE box,
the true box shifted by +-2/4/8/16 px and scaled by +-5/10/20 %, and the box bruterect found.
Raw score S = mean s_k logit_k and centred score (per-bit mean over the image's grid windows subtracted).
    python diag2a.py --method stegastamp
Tells: search failure (true scores higher than found) vs objective failure (found scores higher), and the peak width."""
import argparse, numpy as np, torch
import data
from bruterect import Scorer, grid, SIZES

ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); a = ap.parse_args()
sc = Scorer(a.method); lim = 2.2 if a.method == "trustmark" else 16.5; G = grid(lim, SIZES)
ds = data.load_split("train", [a.method]); df = data.train_table().set_index("name")
z = np.load(data.RESULTS / f"brute_{a.method}_train.npz"); found = dict(zip(z["names"].tolist(), z["box"]))
R = {"raw": [], "cen": []}; shifts = (0, 2, 4, 8, 16); scales = (0.8, 0.9, 0.95, 1.05, 1.1, 1.2)
for i, m in enumerate(ds.meta):
    n = m["image_name"]
    if df.loc[n].regime != "partial" or n not in found:
        continue
    it = ds[i]; g = it["mask"].astype(bool); ys, xs = np.nonzero(g)
    if len(ys) == 0:
        continue
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if g[y0:y1, x0:x1].mean() < 0.95:
        continue
    x = torch.from_numpy(it["image"].copy()).permute(2, 0, 1)[None].float().div(255).cuda()
    s = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).cuda()
    boxes = [(y0, x0, y1, x1)]
    for d in shifts[1:]:
        boxes += [(y0 + d, x0, y1 + d, x1), (y0, x0 + d, y1, x1 + d)]
    cy, cx, h, w = (y0 + y1) / 2, (x0 + x1) / 2, y1 - y0, x1 - x0
    for f in scales:
        boxes.append((int(cy - h * f / 2), int(cx - w * f / 2), int(cy + h * f / 2), int(cx + w * f / 2)))
    boxes.append(tuple(int(v) for v in found[n]))
    boxes = [(max(0, b0), max(0, b1), min(512, b2), min(512, b3)) for b0, b1, b2, b3 in boxes]
    sc.bias = None; raw, _ = sc.score(x, boxes, s)
    lg = sc.logits_all(x, G); sc.bias = lg.mean(0); cen, _ = sc.score(x, boxes, s); sc.bias = None
    for k, v in (("raw", raw), ("cen", cen)):
        R[k].append(v)
nsh = 2 * (len(shifts) - 1)
for k, V in R.items():
    V = np.array(V); true, fnd = V[:, 0], V[:, -1]
    print(f"{a.method} {k}: n={len(V)}  true box > found box in {np.mean(true > fnd):.0%} of images  (median true {np.median(true):.2f}, found {np.median(fnd):.2f})")
    for j, d in enumerate(shifts[1:]):
        print(f"   shift {d:2d}px: median score / true = {np.median(V[:, 1 + 2 * j:3 + 2 * j].mean(1) / np.maximum(true, 1e-6)):.2f}")
    for j, f in enumerate(scales):
        print(f"   scale x{f}: median score / true = {np.median(V[:, 1 + nsh + j] / np.maximum(true, 1e-6)):.2f}")
