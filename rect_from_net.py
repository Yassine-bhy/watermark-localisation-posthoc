"""Rectangle pipeline on partial images: network map -> candidate boxes -> decoder refinement -> keep if it decodes.
    python rect_from_net.py --method trustmark --split train --preds wam256_ft      (held-out CV maps)
    python rect_from_net.py --method trustmark --split test  --preds wam256_ft_all_test
Candidates: bounding boxes of the largest connected regions of (map > t) for several t, plus the box of the top-q
cells; each is refined with rectsearch.refine (decoder agreement with the known message).
Writes results/rect_<preds>_<method>_<split>.npz: names, best box, best agreement for every image."""
import argparse
import time

import numpy as np
import torch
from scipy import ndimage as ndi

import data
from rectsearch import refine
from wmloc.watermarks import load_watermark


def candidates(p, k=6):
    out = []
    for t in (0.3, 0.5, 0.7):
        lab, n = ndi.label(p > t)
        if n:
            sizes = ndi.sum(np.ones_like(p), lab, range(1, n + 1))
            for j in np.argsort(sizes)[::-1][:2]:
                ys, xs = np.nonzero(lab == j + 1)
                if len(ys) >= 4:
                    out.append((ys.min() * 4, xs.min() * 4, (ys.max() + 1) * 4, (xs.max() + 1) * 4))
    for q in (0.97, 0.9):
        ys, xs = np.nonzero(p >= np.quantile(p, q))
        out.append((ys.min() * 4, xs.min() * 4, (ys.max() + 1) * 4, (xs.max() + 1) * 4))
    uniq = []
    for b in out:                                   # drop near-duplicates
        if all(max(abs(np.array(b) - np.array(u))) > 16 for u in uniq):
            uniq.append(b)
    return uniq[:k]


def load_preds(tag):
    P = {}
    if tag.endswith("_test"):
        z = np.load(data.RESULTS / "preds" / f"{tag}.npz"); P.update(zip(z["names"].tolist(), z["prob"].astype(np.float32)))
    else:
        for k in range(5):
            z = np.load(data.RESULTS / "preds" / f"{tag}_f{k}.npz"); P.update(zip(z["names"].tolist(), z["prob"].astype(np.float32)))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--split", default="train")
    ap.add_argument("--preds", default="wam256_ft"); ap.add_argument("--only_partial", type=int, default=1)
    a = ap.parse_args()
    P = load_preds(a.preds)
    wm = load_watermark(a.method, models_dir=data.TASK_DIR / "models", device="cuda")
    ds = data.load_split(a.split, [a.method])
    gs = dict(zip(*[np.load(data.RESULTS / "signals" / f"{a.method}_{a.split}.npz")[k].tolist() for k in ("names", "global_soft")]))
    asp = 2.2 if a.method == "trustmark" else 16.5
    names, boxes, agrees, t0 = [], [], [], time.time()
    for i, m in enumerate(ds.meta):
        n = m["image_name"]
        if a.only_partial and gs[n] > 0.4:          # the gate says "full": no rectangle search
            continue
        it = ds[i]
        x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).cuda()
        s = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).cuda()
        best = ((0, 0, 512, 512), -1.0)
        for c in candidates(P[n]):
            b, ag = refine(wm, x, s, c, n_samples=128, spread=0.35, max_aspect=asp)
            if ag > best[1]:
                best = (b, ag)
            if best[1] > 0.8:
                break
        names.append(n); boxes.append(best[0]); agrees.append(best[1])
        if len(names) % 20 == 0:
            print(f"{len(names)} images {time.time() - t0:.0f}s  found(>0.5) {np.mean(np.array(agrees) > 0.5):.0%}", flush=True)
    np.savez(data.RESULTS / f"rect_{a.preds}_{a.method}_{a.split}.npz", names=np.array(names), box=np.array(boxes), agree=np.array(agrees))
    print("done", len(names), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
