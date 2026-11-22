"""Kill test: TruFor forgery maps (results/trufor_train.npz, already computed for FULL train images) as a hole detector.
Pass threshold set in advance: median within-image AUC (hole = positive) >= 0.70 for at least one method,
or >= +0.01 full-regime mIoU when averaged with our map. CPU only, < 2 min."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, data
from diagnostics import auc, miou_batch
from evalmaps import load_maps
TAGS = {"pixelseal": "ens3_pixelseal", "trustmark": "ens2_trustmark", "stegastamp": "ens2_stegastamp"}
z = np.load(data.RESULTS / "trufor_train.npz"); T = dict(zip(z["names"].tolist(), z["prob"].astype(np.float32)))
df = data.train_table().set_index("name"); g = data.train_masks128()
TS = np.linspace(0.02, 0.98, 49)
for m, tag in TAGS.items():
    names, P = load_maps(tag, "train"); sel = [i for i, n in enumerate(names) if n in T and df.loc[n].full_watermark and not g[n].all()]
    G = np.stack([g[names[i]] for i in sel]).astype(bool); Pm = P[sel]; Tm = np.stack([T[names[i]] for i in sel])
    a_t = np.median([auc(Tm[i].ravel(), ~G[i].ravel()) for i in range(len(sel))]); a_p = np.median([auc(-Pm[i].ravel(), ~G[i].ravel()) for i in range(len(sel))])
    rk = lambda X: np.stack([x.ravel().argsort().argsort().reshape(x.shape) / x.size for x in X])
    best = lambda X: max(miou_batch(X > t, G).mean() for t in TS)
    mix = {w: best(Pm - w * Tm) for w in (0.25, 0.5, 1.0)}
    print(f"{m:11s} full n={len(sel)}  hole-AUC trufor {a_t:.3f} | our map {a_p:.3f} | mIoU@best-thr ours {best(Pm):.3f} | ours - w*trufor {({k: round(v, 3) for k, v in mix.items()})} | trufor alone {best(1 - Tm):.3f}")
