"""Check (CPU, seconds): did the real images pass through the FLUX VAE?  Uses results/aerr/flux_train.npz
(|x - D(E(x))| with the FLUX VAE, already computed by aerr.py, x in [-1, 1]).
Reference from vae_test.py log: a FIRST FLUX round trip changes a clean COCO image by ~3.0-3.6 / 255 (mean abs).
An image that is already a FLUX-VAE output should change much less on a second round trip.
Also: error inside the minority region vs outside (full: hole vs watermarked; partial: watermarked vs rest)."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, data
z = np.load(data.RESULTS / "aerr" / "flux_train.npz"); E = dict(zip(z["names"].tolist(), z["l1"].astype(np.float32) * 127.5))
df = data.train_table().set_index("name"); g = data.train_masks128()
for m in data.METHODS:
    for reg in ("full", "partial"):
        ns = [n for n in df.index if df.loc[n].method == m and df.loc[n].regime == reg and 0 < g[n].mean() < 1]
        tot = np.array([E[n].mean() for n in ns]); wmk = np.array([E[n][g[n] > 0].mean() for n in ns]); cln = np.array([E[n][g[n] == 0].mean() for n in ns])
        print(f"{m:11s} {reg:8s} n={len(ns):3d}  2nd-round-trip change: median {np.median(tot):.2f}/255 (q10 {np.quantile(tot,.1):.2f}, q90 {np.quantile(tot,.9):.2f}) | watermarked area {np.median(wmk):.2f} vs clean area {np.median(cln):.2f} | share of images where clean > watermarked: {np.mean(cln > wmk):.2f}")
