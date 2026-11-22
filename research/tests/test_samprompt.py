"""Kill test (1 GPU, ~5 min): SAM (ViT-B, external/sam) prompted by the peak of OUR map, to fix the SHAPE.
Why: transplanted objects have map AUC 0.94-0.98 but mIoU 0.66-0.72, full images AUC 0.90-0.93 but mIoU 0.60-0.67.
Partial image: point prompt at the max of the smoothed map; full image: at the min (the hole). SAM returns 3 masks;
'pick' = the one with the largest contrast (mean map inside - mean map in a 6-cell ring outside), sign flipped for holes.
Variants: base (best global threshold, in-sample = optimistic baseline), sam (always), agree (SAM only if its IoU with
the thresholded map component under the prompt is >= 0.3), oracle3 (best of the 3 masks), oracle (max of base, oracle3).
True regime is used (no gate). Pass threshold (set in advance): 'agree' beats 'base' by >= 0.03 in some regime x method."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F
from scipy import ndimage as ndi
from segment_anything import SamPredictor, sam_model_registry
import data
from diagnostics import miou_batch
from evalmaps import load_maps
TAGS = {"pixelseal": "ens3_pixelseal", "trustmark": "ens2_trustmark", "stegastamp": "ens2_stegastamp"}
sam = sam_model_registry["vit_b"](checkpoint=str(data.HERE / "external" / "sam" / "sam_vit_b_01ec64.pth")).cuda().eval(); pred = SamPredictor(sam)
df = data.train_table().set_index("name"); g = data.train_masks128(); TS = np.linspace(0.04, 0.96, 24)
down = lambda m: (F.avg_pool2d(torch.from_numpy(m.astype(np.float32))[None, None], 4)[0, 0] > 0.5).numpy()
sc = lambda a, b: float(miou_batch(a[None], b[None])[0])
for meth, tag in TAGS.items():
    names, P = load_maps(tag, "train"); P = dict(zip(names, P)); ds = data.load_split("train", [meth]); rows = []
    G = {n: g[n].astype(bool) for n in names}; full = {n: bool(df.loc[n].full_watermark) for n in names}
    thr = {}
    for reg in (True, False):
        ns = [n for n in names if full[n] == reg]; thr[reg] = TS[int(np.argmax([np.mean([sc(P[n] > t, G[n]) for n in ns]) for t in TS]))]
    for i in range(len(ds)):
        it = ds[i]; n = it["image_name"]; p = P[n]; m = G[n]; fl = full[n]; q = ndi.gaussian_filter(p, 1.5)
        base_mask = p > thr[fl]; s = -q if fl else q; y, x = np.unravel_index(np.argmax(s), s.shape)
        pred.set_image(it["image"])
        with torch.no_grad():
            ms, _, _ = pred.predict(point_coords=np.array([[x * 4 + 2, y * 4 + 2]]), point_labels=np.array([1]), multimask_output=True)
        cands = [down(k) for k in ms]; cands = [c for c in cands if c.any() and not c.all()] or [np.zeros((128, 128), bool)]
        def contrast(c):
            ring = ndi.binary_dilation(c, iterations=6) & ~c
            return (s[c].mean() - s[ring].mean()) if c.any() and ring.any() else -9
        pick = max(cands, key=contrast); tomask = (lambda c: ~c) if fl else (lambda c: c)
        region = (~base_mask) if fl else base_mask; lab, _ = ndi.label(region); comp = (lab == lab[y, x]) if lab[y, x] else np.zeros_like(region)
        iou = (pick & comp).sum() / max((pick | comp).sum(), 1)
        b = sc(base_mask, m); sm = sc(tomask(pick), m); o3 = max(sc(tomask(c), m) for c in cands)
        kind = "full" if fl else ("rect" if m.sum() / ((np.nonzero(m)[0].max() - np.nonzero(m)[0].min() + 1) * (np.nonzero(m)[1].max() - np.nonzero(m)[1].min() + 1)) > 0.98 else "object")
        rows.append((kind, b, sm, sm if iou >= 0.3 else b, o3, max(b, o3), iou >= 0.3))
    for kind in ("full", "object", "rect"):
        R = np.array([r[1:] for r in rows if r[0] == kind], float)
        print(f"{meth:11s} {kind:6s} n={len(R):3d}  base {R[:,0].mean():.3f} | sam {R[:,1].mean():.3f} | agree {R[:,2].mean():.3f} (used {R[:,5].mean():.2f}) | oracle3 {R[:,3].mean():.3f} | oracle {R[:,4].mean():.3f}", flush=True)
