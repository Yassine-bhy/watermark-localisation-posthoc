"""Round 2 (full images only), CPU, read-only, ~3 min.
A. Synthetic-to-real gap: zero-shot (synthetic-only, '<tag>pre') vs fine-tuned maps on real FULL train images.
B. Structure of the true hole masks at 512 px: grid alignment of edges (8 / 16 px), blockiness, border contact, position.
C. Component oracle: candidates = connected components of (smoothed map < t) for a ladder of thresholds; greedy oracle
   subset by true mIoU.  Ceiling of ANY component-rescoring / multi-hole selection rule on the current maps."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, data
from PIL import Image
from scipy import ndimage as ndi
from diagnostics import auc, miou_batch
from evalmaps import load_maps
df = data.train_table().set_index("name"); g = data.train_masks128()
TS = np.linspace(0.04, 0.96, 24)
print("== A. zero-shot (synthetic only) vs fine-tuned, FULL images: median hole AUC | mIoU at best global threshold")
for m in data.METHODS:
    for tag in (f"e2eR512pre_{m}", f"e2eR512_{m}", f"e2efluxpre_{m}", f"e2eflux_{m}"):
        try: names, P = load_maps(tag, "train")
        except Exception: continue
        sel = [i for i, n in enumerate(names) if df.loc[n].full_watermark]; G = np.stack([g[names[i]] for i in sel]).astype(bool); Q = P[sel]
        a = np.median([auc(-Q[i].ravel(), ~G[i].ravel()) for i in range(len(sel))])
        print(f"   {tag:26s} AUC {a:.3f} | mIoU {max(miou_batch(Q > t, G).mean() for t in TS):.3f}")
print("== B. true hole masks (512 px) of full images")
mdir = data.DATA / "train" / "masks"; rows = []; pos = np.zeros((128, 128))
for n in df.index[df.full_watermark]:
    M = np.asarray(Image.open(mdir / f"{n}.png")); M = (M if M.ndim == 2 else M[..., 0]) > 127; H = ~M
    vals = np.unique(np.asarray(Image.open(mdir / f"{n}.png")))
    ex = np.nonzero(np.diff(H.astype(np.int8), axis=1))[1] + 1; ey = np.nonzero(np.diff(H.astype(np.int8), axis=0))[0] + 1
    e = np.r_[ex, ey]
    b8 = ndi.zoom(H.reshape(64, 8, 64, 8).mean((1, 3)) > 0.5, 8, order=0)              # hole snapped to the 8 px grid
    border = H[0].any() or H[-1].any() or H[:, 0].any() or H[:, -1].any()
    lab, k = ndi.label(H); er = ndi.binary_erosion(H, iterations=2).sum() / max(H.sum(), 1)
    rows.append((np.mean(e % 8 == 0) if len(e) else np.nan, np.mean(e % 16 == 0) if len(e) else np.nan, (b8 == H).mean(), border, k, er, len(vals)))
    pos += g[n] == 0
R = np.array(rows, float)
print(f"   edge positions multiple of 8: {np.nanmean(R[:,0]):.3f} (chance 0.125) | of 16: {np.nanmean(R[:,1]):.3f} (chance 0.0625)")
print(f"   mask identical to its 8-px-grid version: {np.mean(R[:,2] == 1):.2f} of images | touches image border: {R[:,3].mean():.2f} | components@512 median {np.median(R[:,4]):.0f} | area left after 2-px erosion: {np.median(R[:,5]):.2f} | grey levels in mask png: {np.median(R[:,6]):.0f}")
pp = pos / len(R); print(f"   position prior: P(hole) centre 64x64 block {pp[32:96,32:96].mean():.3f} vs outer ring {(pp.sum()-pp[32:96,32:96].sum())/(128*128-64*64):.3f}; max cell {pp.max():.2f}")
print("== C. component oracle on current maps (full images)")
for m, tag in (("pixelseal", "ens3_pixelseal"), ("trustmark", "ens2_trustmark"), ("stegastamp", "ens2_stegastamp")):
    names, P = load_maps(tag, "train"); sel = [i for i, n in enumerate(names) if df.loc[n].full_watermark]
    G = np.stack([g[names[i]] for i in sel]).astype(bool); Q = P[sel]; base_t = max(TS, key=lambda t: miou_batch(Q > t, G).mean()); base = miou_batch(Q > base_t, G)
    orc = []; hit = []
    for q, gt in zip(Q, G):
        qs = ndi.gaussian_filter(q, 1.5); cands = []
        for t in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
            lab, k = ndi.label(qs < t)
            cands += [lab == j for j in range(1, k + 1) if 4 <= (lab == j).sum() <= 0.6 * q.size]
        cur = np.ones_like(gt); best = miou_batch(cur[None], gt[None])[0]
        for _ in range(4):
            sc = [miou_batch((cur & ~c)[None], gt[None])[0] for c in cands]
            if not sc or max(sc) <= best + 1e-6: break
            best = max(sc); cur = cur & ~cands[int(np.argmax(sc))]
        orc.append(best)
        hl, hk = ndi.label(~gt); hit.append(np.mean([any((c & (hl == j)).sum() > 0.3 * (hl == j).sum() and (c & (hl == j)).sum() > 0.3 * c.sum() for c in cands) for j in range(1, hk + 1)]))
    print(f"   {m:11s} threshold baseline {base.mean():.3f} | greedy component oracle (<=4 comps) {np.mean(orc):.3f} | share of true holes with a matching candidate component: {np.mean(hit):.2f}")
