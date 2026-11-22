"""EDA: is the full-image failure driven by image texture? (watermark residuals are often content-adaptive)
For full train images: per 4x4 cell, local texture = mean |Laplacian| of gray image (cell neighbourhood 16 px).
Report: within-image AUC of the map for hole detection, split by texture tercile of the cell;
and hole-component detection rate vs mean texture of the hole."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); import data
from scipy.stats import rankdata
def roc_auc_score(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
tab = data.train_table(); g = data.train_masks128()
tags = {"pixelseal": "ens3", "trustmark": "ens4", "stegastamp": "ens4"}
for m, t in tags.items():
    z = np.load(f"results/preds/{t}_{m}_train.npz"); maps = dict(zip(z["names"].tolist(), z["prob"]))
    ds = data.load_split("train", [m]); rows = []
    for i in range(len(ds)):
        it = ds[i]
        if not it["full_watermark"]: continue
        x = torch.from_numpy(it["image"].astype(np.float32)).mean(-1)[None, None]
        lap = (F.conv2d(x, torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], dtype=torch.float32), padding=1)).abs()
        tex = F.avg_pool2d(lap, 16, stride=4, padding=6)[0, 0, :128, :128].numpy()
        tex = np.log1p(tex); P = np.asarray(maps[it["image_name"]], np.float32); P = P if P.shape == (128,128) else np.array(torch.nn.functional.interpolate(torch.tensor(P)[None,None].float(), 128)[0,0])
        G = g[it["image_name"]]; rows.append((tex.ravel(), P.ravel(), 1 - G.ravel()))
    T = np.concatenate([r[0] for r in rows]); P = np.concatenate([r[1] for r in rows]); H = np.concatenate([r[2] for r in rows])
    q = np.quantile(T, [1/3, 2/3]); bins = np.digitize(T, q)
    out = []
    for b in range(3):
        s = bins == b; out.append((round(H[s].mean(), 3), round(roc_auc_score(H[s], -P[s]), 3), round(P[s & (H == 0)].mean(), 3), round(P[s & (H == 1)].mean(), 3)))
    print(m, "texture terciles low/mid/high: (hole frac, AUC hole-vs-wm, mean map in wm, mean map in hole)", out, flush=True)
