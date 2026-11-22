"""Kill test (CPU, <1 min): can our maps SEED a dense local box search on rectangle (outpainting) images?
Per rectangle image: within-image AUC, and whether one of the top-k peaks of the smoothed map (non-max suppression
radius 6 cells) falls inside the true rectangle (dilated by 1 cell). Also for object (transplant) images.
Pass threshold (set in advance): top-5 hit rate >= 0.70 on rectangles for PixelSeal or StegaStamp."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, data
from scipy import ndimage as ndi
from diagnostics import auc
from evalmaps import load_maps
TAGS = {"pixelseal": ["ens3_pixelseal", "e2eR512_pixelseal"], "trustmark": ["ens2_trustmark", "e2eR512_trustmark"], "stegastamp": ["ens2_stegastamp", "e2eR512_stegastamp"]}
df = data.train_table().set_index("name"); g = data.train_masks128()
def peaks(p, k, r=6):
    q = ndi.gaussian_filter(p, 1.0).copy(); out = []
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(q), q.shape); out.append((y, x)); q[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = -1
    return out
for meth, tags in TAGS.items():
    for tag in tags:
        names, P = load_maps(tag, "train"); R = {"rect": [], "object": []}
        for i, n in enumerate(names):
            if df.loc[n].full_watermark: continue
            m = g[n].astype(bool); ys, xs = np.nonzero(m)
            kind = "rect" if m.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) > 0.98 else "object"
            md = ndi.binary_dilation(m, iterations=1); pk = peaks(P[i], 10)
            hit = [any(md[y, x] for y, x in pk[:k]) for k in (1, 3, 5, 10)]
            R[kind].append([auc(P[i].ravel(), m.ravel()), m.mean()] + hit)
        for kind, v in R.items():
            v = np.array(v, float); small = v[:, 1] < 0.02
            print(f"{tag:22s} {kind:6s} n={len(v):2d} AUC median {np.median(v[:,0]):.3f} (q25 {np.quantile(v[:,0],.25):.3f}) | peak-in-region top1/3/5/10: " + " ".join(f"{x:.2f}" for x in v[:, 2:].mean(0)) + f" | area<2% (n={small.sum()}): top5 {v[small, 4].mean():.2f}")
