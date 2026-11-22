"""Kill test (1 GPU, ~3 min TrustMark / ~7 min StegaStamp): ZOOMED tile inference of the e2e segmenter.
Hypothesis: outpainting rectangles hold a watermark that was SHRUNK with the original image (factor ~3-6), so at
native 512 px its texture is at the wrong scale for the decoder's early layers (our maps: AUC ~0.5-0.75 on rectangles vs
0.94-0.98 on transplanted objects, research/tests/test_seedhit.py). Decoding the up-scaled true crop works (TM 84 %,
SS 64 %), so the signal is there once magnified. Test: run the same network on tiles of side 512/z (the network
resizes every tile to 512), stitch on the 128 grid, compare z = 1 with z = 2, 3, 4, 6 and max / mean over zooms.
Protocol: stage A on synth_flux (3000 steps, res 512) + fine-tune on real folds 0-2; evaluate on folds 3-4 only.
Pass threshold (set in advance): on held-out RECTANGLE images, median within-image AUC +0.10 or top-5 peak hit
rate >= 0.60 for some zoom setting, for at least one method.   Writes nothing but its stdout."""
import sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F
from scipy import ndimage as ndi
import data, e2e
from diagnostics import auc
from wmloc.watermarks import load_watermark

method = sys.argv[1]; t0 = time.time(); torch.manual_seed(0)
e2e.OUT = 64; e2e.DEEP = False; e2e.FLIP = False
wm = load_watermark(method, models_dir=data.TASK_DIR / "models", device="cuda"); net = e2e.Net(wm).cuda().float(); net.res = 512
tr = data.load_split("train", [method]); names = [tr[i]["image_name"] for i in range(len(tr))]
Xr = [tr[i]["image"] for i in range(len(tr))]; g = data.train_masks128(); Mr = [g[n] for n in names]
df = data.train_table().set_index("name").loc[names]; fold = df.fold.values; full = df.full_watermark.values
sh = sorted(p for p in (data.RESULTS / "synth_flux").glob(f"{method}_*") if (p / "images.npy").exists())
Xs = [x for s in sh for x in np.load(s / "images.npy", mmap_mode="r")]; Ms = [m for s in sh for m in np.load(s / "masks.npy")]
net = e2e.train(net, Xs, Ms, 3000, 1e-3, 16, 0); print("stage A", f"{time.time() - t0:.0f}s", flush=True)
idx = np.nonzero(fold <= 2)[0]; e2e.train(net, [Xr[i] for i in idx], [Mr[i] for i in idx], 30 * len(idx) // 16, 3e-4, 16, 1)
print("fine-tuned", f"{time.time() - t0:.0f}s", flush=True); net.eval()

@torch.no_grad()
def zoom_map(img, z):
    x = torch.from_numpy(img.copy()).cuda().permute(2, 0, 1)[None].float() / 255
    if z == 1:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.sigmoid(net(x).float())[0].cpu().numpy().repeat(2, 0).repeat(2, 1)
    T = int(round(512 / z / 8)) * 8; st = T // 2; pos = sorted(set(list(range(0, 512 - T + 1, st)) + [512 - T]))
    tiles = torch.cat([x[:, :, y:y + T, xx:xx + T] for y in pos for xx in pos]); out = []
    for i in range(0, len(tiles), 16):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(net(tiles[i:i + 16]).float()))
    P = F.interpolate(torch.cat(out)[:, None], size=(T // 4, T // 4), mode="area")[:, 0].cpu().numpy()
    acc = np.zeros((128, 128), np.float32); cnt = np.zeros((128, 128), np.float32); k = 0
    for y in pos:
        for xx in pos:
            acc[y // 4:y // 4 + T // 4, xx // 4:xx // 4 + T // 4] += P[k]; cnt[y // 4:y // 4 + T // 4, xx // 4:xx // 4 + T // 4] += 1; k += 1
    return acc / np.maximum(cnt, 1)

def peaks_hit(p, m, k=5, r=6):
    q = ndi.gaussian_filter(p, 1.0).copy(); md = ndi.binary_dilation(m, iterations=1)
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(q), q.shape)
        if md[y, x]: return 1.0
        q[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = -1
    return 0.0

ZS = (1, 2, 3, 4, 6); te = np.nonzero(fold >= 3)[0]; rows = []
for i in te:
    m = Mr[i].astype(bool); maps = {z: zoom_map(Xr[i], z) for z in ZS}
    maps["max"] = np.max([maps[z] for z in ZS], 0); maps["mean"] = np.mean([maps[z] for z in ZS], 0); maps["max1-3"] = np.max([maps[z] for z in (1, 2, 3)], 0)
    if full[i]: kind = "full"
    else:
        ys, xs = np.nonzero(m); kind = "rect" if m.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) > 0.98 else "object"
    rows.append((kind, {k: (auc(v.ravel(), m.ravel()), peaks_hit(v, m) if kind != "full" else np.nan) for k, v in maps.items()}))
print("zoom tiles done", f"{time.time() - t0:.0f}s")
for kind in ("rect", "object", "full"):
    R = [r for k, r in rows if k == kind]
    print(f"{method} {kind:6s} n={len(R)}")
    for k in list(ZS) + ["max1-3", "max", "mean"]:
        a = np.array([r[k][0] for r in R]); h = np.array([r[k][1] for r in R])
        print(f"   zoom {str(k):6s} AUC median {np.median(a):.3f}  q25 {np.quantile(a, .25):.3f}   top-5 peak hit {np.nanmean(h) if kind != 'full' else float('nan'):.2f}")
