"""Does TrustMark's own watermark REMOVER localise? d = x - remover(x) at 256; local energy of d vs ground truth."""
import numpy as np, torch, torch.nn.functional as F, sys
import data
from diagnostics import auc
from wmloc.watermarks import load_watermark
dev = "cuda" if torch.cuda.is_available() else "cpu"
wm = load_watermark("trustmark", models_dir=data.TASK_DIR / "models", device=dev); rm = wm.tm.removal
ds = data.load_split("train", ["trustmark"]); df = data.train_table().set_index("name")
res = {"full": [], "partial": []}
lim = int(sys.argv[1]) if len(sys.argv) > 1 else len(ds)
for i in range(lim):
    it = ds[i]; x = torch.from_numpy(it["image"].copy()).permute(2, 0, 1)[None].float().div(255).to(dev)
    s = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    with torch.no_grad():
        d = (rm(s).clamp(-1, 1) - s)
    d = d - d.mean((2, 3), keepdim=True)
    e = F.avg_pool2d((d ** 2).sum(1, keepdim=True), 2)          # 128 x 128 energy
    for sg, k in ((0, 1),):
        pass
    e = F.avg_pool2d(F.pad(e, (3, 3, 3, 3), mode="reflect"), 7, 1)[0, 0].cpu().numpy()
    g = it["mask128"].astype(bool)
    if 0 < g.mean() < 1:
        res[df.loc[it["image_name"]].regime].append(auc(e.ravel(), g.ravel()))
for k, v in res.items():
    print(k, "n", len(v), "median within-image AUC of remover energy:", np.median(v).round(3))
