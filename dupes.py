"""Do source photos repeat across the dataset? Thumbnail (32x32 gray, normalised) correlation between all images."""
import numpy as np, torch, torch.nn.functional as F
import data
names, th, split = [], [], []
for sp in ("train", "test"):
    ds = data.load_split(sp)
    for i in range(len(ds)):
        it = ds[i]; x = torch.from_numpy(it["image"].copy()).float().mean(-1)[None, None]
        t = F.adaptive_avg_pool2d(x, 32).flatten(); t = (t - t.mean()) / (t.std() + 1e-6)
        names.append(it["image_name"]); th.append(t.numpy()); split.append(sp)
T = np.stack(th); C = T @ T.T / T.shape[1]; np.fill_diagonal(C, -1)
best = C.max(1); arg = C.argmax(1)
print("n", len(T), "best-corr quantiles", np.quantile(best, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3))
print("frac with best>0.9:", (best > 0.9).mean().round(3), " >0.8:", (best > 0.8).mean().round(3))
for i in np.argsort(-best)[:10]:
    print(names[i], split[i], "<->", names[arg[i]], split[arg[i]], round(best[i], 3))
np.savez("results/dupes.npz", names=np.array(names), best=best, arg=arg)
