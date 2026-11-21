"""Full images: does the image's fine detail correlate LOCALLY with the expected, message-specific watermark pattern?
t = Emb(y, m) - mean_j Emb(y, m'_j)  (computed on the observed image y; for full images the rectangle = whole image).
Local normalised correlation of high-pass(y) with high-pass(t) in w x w windows -> 128 map; within-image AUC vs GT."""
import sys, numpy as np, torch, torch.nn.functional as F
import data
from diagnostics import auc
from wmloc.watermarks import load_watermark
torch.set_num_threads(32)
method = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"; N = int(sys.argv[3]) if len(sys.argv) > 3 else 25
wm = load_watermark(method, models_dir=data.TASK_DIR / "models", device=dev)
ds = data.load_split("train", [method]); df = data.train_table().set_index("name"); rng = np.random.default_rng(0)
T = lambda a: torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1)[None]
hp = lambda t: t - F.avg_pool2d(F.pad(t, (2, 2, 2, 2), mode="reflect"), 5, 1)
box = lambda t, k: F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="reflect"), k, 1)
res = {k: [] for k in (9, 17, 33)}; n = 0
for i in range(len(ds)):
    it = ds[i]
    if df.loc[it["image_name"]].regime != "full" or it["mask128"].all():
        continue
    y = it["image"]; m = it["message"]
    e = T(wm.embed(y, m)); r = np.mean([T(wm.embed(y, rng.integers(0, 2, wm.nbits))).numpy() for _ in range(4)], 0)
    t = hp(e - torch.from_numpy(r)); h = hp(T(y))
    for k in res:
        c = box((h * t).sum(1, keepdim=True), k) / (torch.sqrt(box((h * h).sum(1, keepdim=True), k) * box((t * t).sum(1, keepdim=True), k)) + 1e-8)
        c = F.avg_pool2d(c, 4)[0, 0].numpy()
        res[k].append(auc(c.ravel(), it["mask128"].ravel().astype(bool)))
    n += 1
    if n == N:
        break
for k, v in res.items():
    print(method, f"window {k:2d}px: median within-image AUC {np.median(v):.3f}  (n={len(v)})")
