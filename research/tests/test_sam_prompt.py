"""Kill test (1 GPU, ~5-10 min): SAM (ViT-B, external/sam) prompted by the peaks of OUR minority-class map.
Idea: our maps locate the minority region (within-image AUC 0.76-0.93) but are blurry; holes / transplanted regions are
often objects, so an object mask at the peak could give the shape.
Per image: q = 1 - p (full regime) or p (partial); top-3 peaks of the smoothed q -> 3 point prompts x 3 SAM masks.
'selected' = candidate maximising sum over its cells of (q - tau)  (tau in-sample per method x regime, like the
baseline threshold); 'oracle' = best candidate by true mIoU (ceiling of any selection rule);
'hybrid' = selected SAM mask if its criterion beats the thresholded map's own criterion, else the thresholded map.
Regime = TRUE regime (kill test).  Pass threshold (set in advance): selected or hybrid >= baseline + 0.03 in >= 2 of
the 6 cells.  Writes research/out/sam_prompt.npz (candidates) for reuse."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F, data
from scipy import ndimage as ndi
from segment_anything import SamPredictor, sam_model_registry
from diagnostics import miou_batch
from evalmaps import load_maps
TAGS = {"pixelseal": "ens3_pixelseal", "trustmark": "ens2_trustmark", "stegastamp": "ens2_stegastamp"}
sam = sam_model_registry["vit_b"](checkpoint=str(data.HERE / "external" / "sam" / "sam_vit_b_01ec64.pth")).cuda().eval(); pred = SamPredictor(sam)
df = data.train_table().set_index("name"); g = data.train_masks128(); TAUS = (0.3, 0.4, 0.5, 0.6, 0.7)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 6
def peaks(q, k=3, r=12):
    s = ndi.gaussian_filter(q, 2).copy(); out = []
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(s), s.shape); out.append((y, x)); s[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = -1
    return out
def mi(m, gt): return float(miou_batch(m[None], gt[None])[0])
for meth, tag in TAGS.items():
    names, P = load_maps(tag, "train"); P = dict(zip(names, P)); ds = data.load_split("train", [meth]); rows = []
    for i in range(min(len(ds), N)):
        it = ds[i]; n = it["image_name"]; full = bool(df.loc[n].full_watermark); gt = g[n].astype(bool)
        q = 1 - P[n] if full else P[n]; mino = ~gt if full else gt
        pred.set_image(it["image"]); cands = []
        for (y, x) in peaks(q):
            with torch.no_grad():
                ms, _, _ = pred.predict(point_coords=np.array([[x * 4 + 2, y * 4 + 2]]), point_labels=np.array([1]), multimask_output=True)
            cands += [F.avg_pool2d(torch.from_numpy(m[None, None].astype(np.float32)), 4)[0, 0].numpy() > 0.5 for m in ms]
        cands = [c for c in cands if c.any() and c.mean() < 0.6]
        rows.append(dict(full=full, q=q, mino=mino, cands=cands))
    for reg in (True, False):
        R = [r for r in rows if r["full"] == reg]; G = np.stack([r["mino"] for r in R]); Q = np.stack([r["q"] for r in R])
        base_t = max(TAUS + (0.2, 0.8, 0.9), key=lambda t: miou_batch(Q > t, G).mean()); base = miou_batch(Q > base_t, G)
        orc = np.array([max([mi(c, r["mino"]) for c in r["cands"]] + [0.0]) for r in R])
        best = None
        for tau in TAUS:
            sel, hyb = [], []
            for r, b in zip(R, base):
                crit = [(c * (r["q"] - tau)).sum() for c in r["cands"]]
                if not crit: sel.append(b); hyb.append(b); continue
                j = int(np.argmax(crit)); s = mi(r["cands"][j], r["mino"]); sel.append(s)
                hyb.append(s if crit[j] > ((r["q"] > base_t) * (r["q"] - tau)).sum() else b)
            if best is None or max(np.mean(sel), np.mean(hyb)) > max(best[1], best[2]): best = (tau, np.mean(sel), np.mean(hyb))
        print(f"{meth:11s} {'full' if reg else 'partial':8s} n={len(R):3d}  map-threshold baseline {base.mean():.3f} | SAM selected {best[1]:.3f} | hybrid {best[2]:.3f} (tau {best[0]}) | SAM oracle {orc.mean():.3f} | max(oracle, baseline) {np.maximum(orc, base).mean():.3f}", flush=True)
