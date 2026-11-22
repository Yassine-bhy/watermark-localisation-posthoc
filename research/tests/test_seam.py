"""Kill test (CPU, ~2 min): do outpainted rectangles leave a straight seam in the IMAGE itself (independent of the
watermark)?  For every partial train image whose mask is an exact rectangle, compute row / column profiles of
several pixel statistics and ask whether the true rectangle borders stand out.
Statistics per column x (same for rows): mean over the rows y0..y1 of |I(y, x+1) - I(y, x)| (gradient across the
border), and the change of local high-pass energy (sharpness) left vs right of x.
Report: z-score of the profile at the true border columns/rows vs all other positions (robust median/MAD), and the
rank of the true border among all positions.  'blind' variant: profiles over the WHOLE image height (no knowledge of
y0..y1), which is what a detector could actually use.
Pass threshold (set in advance): blind z-score > 3 at >= 50 % of the true borders for at least one method."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, data
from scipy import ndimage as ndi

def zrank(p, pos):
    med = np.median(p); mad = np.median(np.abs(p - med)) * 1.4826 + 1e-9
    out = []
    for q in pos:
        if 2 <= q < len(p) - 2:
            v = p[q - 1:q + 2].max(); out.append(((v - med) / mad, (p > v).mean()))
    return out

df = data.train_table().set_index("name")
for m in data.METHODS:
    ds = data.load_split("train", [m]); R = {k: [] for k in ("grad_orc", "grad_blind", "hp_orc", "hp_blind")}; n = 0
    for i in range(len(ds)):
        it = ds[i]
        if df.loc[it["image_name"]].full_watermark:
            continue
        g = it["mask"].astype(bool); ys, xs = np.nonzero(g)
        if len(ys) == 0: continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if g[y0:y1, x0:x1].mean() < 0.99: continue
        n += 1; I = it["image"].astype(np.float32).mean(-1)
        hp = np.abs(I - ndi.uniform_filter(I, 5))                          # local detail energy
        for ax, (a0, a1, b0, b1) in enumerate(((y0, y1, x0, x1), (x0, x1, y0, y1))):   # ax 0: vertical borders (columns)
            J = I if ax == 0 else I.T; H = hp if ax == 0 else hp.T
            d = np.abs(np.diff(J, axis=1))                                 # |I(.,x+1)-I(.,x)|, border between x and x+1
            # second difference isolates a step from smooth texture
            for tag, rows in (("orc", slice(a0, a1)), ("blind", slice(0, 512))):
                p = np.median(d[rows], 0); R["grad_" + tag] += zrank(p, [b0 - 1, b1 - 1])
                e = H[rows].mean(0); k = 6; c = np.cumsum(np.r_[0, e])
                side = np.array([abs((c[x] - c[x - k]) - (c[x + k] - c[x])) / k if k <= x <= 512 - k else 0 for x in range(513)])
                R["hp_" + tag] += zrank(side, [b0, b1])
    print(f"{m:11s} rectangles n={n}")
    for k, v in R.items():
        v = np.array(v); print(f"   {k:11s} borders={len(v):3d}  median z {np.median(v[:,0]):6.2f}  share z>3: {np.mean(v[:,0]>3):.2f}  share z>6: {np.mean(v[:,0]>6):.2f}  median rank (0=top) {np.median(v[:,1]):.3f}")
