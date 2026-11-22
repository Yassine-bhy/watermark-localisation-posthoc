"""Round 2, 'something simple we overlooked' checks on FULL train images (CPU, ~2 min, read-only).
A. Messages: unique per image? shared between train and test or between methods?  (hidden linkage)
B. JPEG history: if the clean original was a JPEG and holes were restored from it AFTER watermarking, hole blocks keep
   exact JPEG structure (quantised 8x8 DCT, block grid) while watermarked blocks lose it.
   B1 grid detector: mean |I(x) - I(x-1)| for columns x = k mod 8, k = 0..7 (same for rows): ratio max / median.
   B2 per 8x8 block (grid offset 0): number of AC DCT coefficients with |c| < 0.5 (all AC, and the 28 highest);
      within-image AUC, hole = positive.  Pass threshold (set in advance): median AUC >= 0.70 for a method.
C. Pixel statistics: share of pixels exactly equal to the 3x3 median (flatness) in hole vs watermarked area."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, data
from scipy.fft import dctn
from scipy import ndimage as ndi
from diagnostics import auc
df = data.train_table().set_index("name")
print("== A. messages")
tr = json.loads((data.DATA / "train" / "metadata.json").read_text()); te = json.loads((data.DATA / "test" / "metadata.json").read_text())
key = lambda m: (m["method"], str(m["message"]))
ktr = [key(m) for m in tr]; kte = [key(m) for m in te]
print(f"   train: {len(ktr)} images, {len(set(ktr))} distinct (method, message) | test: {len(kte)} images, {len(set(kte))} distinct | shared train/test: {len(set(ktr) & set(kte))}")
bits = {m["image_name"]: np.array([int(c) for c in (m["message"] if isinstance(m["message"], str) else "".join(map(str, m["message"])))]) for m in tr}
for meth in data.METHODS:
    B = np.stack([bits[n] for n in df.index[df.method == meth]]); print(f"   {meth}: {B.shape[1]} bits, mean bit {B.mean():.3f}, per-position mean min/max {B.mean(0).min():.2f}/{B.mean(0).max():.2f}")
print("== B/C. JPEG structure and flatness, full images")
hf = np.add.outer(np.arange(8), np.arange(8)) >= 8
for meth in data.METHODS:
    ds = data.load_split("train", [meth]); ratios, a_all, a_hf, flat = [], [], [], []
    for i in range(len(ds)):
        it = ds[i]
        if not it["full_watermark"]: continue
        I = it["image"].astype(np.float32); Y = 0.299 * I[..., 0] + 0.587 * I[..., 1] + 0.114 * I[..., 2]; hole = it["mask"] == 0
        dx = np.abs(np.diff(Y, axis=1)).mean(0); dy = np.abs(np.diff(Y, axis=0)).mean(1)          # dx[j]: between columns j and j+1
        ex = np.array([dx[(np.arange(511) + 1) % 8 == k].mean() for k in range(8)]); ey = np.array([dy[(np.arange(511) + 1) % 8 == k].mean() for k in range(8)])
        ratios.append((ex.max() / np.median(ex), ey.max() / np.median(ey), int(ex.argmax()), int(ey.argmax())))
        Bk = (Y - 128).reshape(64, 8, 64, 8).transpose(0, 2, 1, 3); C = dctn(Bk, axes=(2, 3), norm="ortho"); small = np.abs(C) < 0.5
        z_all = small.reshape(64, 64, 64)[:, :, 1:].sum(-1); z_hf = small[:, :, hf].sum(-1)
        hb = hole.reshape(64, 8, 64, 8).mean((1, 3)) > 0.5
        if 0 < hb.sum() < hb.size:
            a_all.append(auc(z_all.ravel().astype(float), hb.ravel())); a_hf.append(auc(z_hf.ravel().astype(float), hb.ravel()))
        eq = (it["image"].astype(np.int16) == ndi.median_filter(it["image"], size=(3, 3, 1)).astype(np.int16)).all(-1)
        if hole.any(): flat.append((eq[hole].mean(), eq[~hole].mean()))
    R = np.array(ratios); F = np.array(flat)
    print(f"   {meth:11s} grid ratio max/median: columns {np.median(R[:,0]):.3f}, rows {np.median(R[:,1]):.3f} (1.00 = no 8-px grid); argmax k most common: {np.bincount(R[:,2].astype(int), minlength=8).argmax()} ({np.bincount(R[:,2].astype(int), minlength=8).max()}/{len(R)})")
    print(f"   {meth:11s} block DCT near-zero count, hole AUC: all AC {np.median(a_all):.3f} | highest 28 {np.median(a_hf):.3f} (n={len(a_all)}) | pixels equal to 3x3 median: hole {np.median(F[:,0]):.3f} vs watermarked {np.median(F[:,1]):.3f}")
