"""Kill test (CPU, ~5 min): learned rescoring of hole CANDIDATE COMPONENTS on full images (multi-hole selection).
Candidates: connected components of (smoothed map < t) for t in a ladder (nested blobs at several sizes).
Features: area, mean / min / max of the map inside, mean in a 4-cell ring outside, contrast, threshold level, border
contact, bbox fill, number of candidates in the image, image-level map mean and 5 % quantile.
Target: IoU of the candidate with the best-matching true hole component.  Model: small MLP (torch, 2 x 32 hidden, 400 Adam steps; sklearn is not in the venv), fit on
the 4 training folds (same folds as everywhere), applied to the held-out fold.
Selection: sort candidates by predicted IoU, keep those above tau that do not overlap an already kept one
(tau and max count chosen on the training folds).  Final mask = all ones minus kept components.
Baseline: best global threshold fit on the training folds (same CV).  True regime (full images only).
Pass threshold (set in advance): >= +0.02 mean over the three methods, and no method below baseline."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, data
from scipy import ndimage as ndi
import torch
from diagnostics import miou_batch
from evalmaps import load_maps
df = data.train_table().set_index("name"); g = data.train_masks128(); TS = np.linspace(0.04, 0.96, 24)
LAD = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
def cands(q):
    qs = ndi.gaussian_filter(q, 1.5); out = []; seen = set()
    for t in LAD:
        lab, k = ndi.label(qs < t)
        for j in range(1, k + 1):
            c = lab == j; a = int(c.sum())
            if a < 4 or a > 0.6 * q.size: continue
            key = (a, int(np.flatnonzero(c)[0]))
            if key in seen: continue
            seen.add(key); ring = ndi.binary_dilation(c, iterations=4) & ~c; ys, xs = np.nonzero(c)
            f = [a / q.size, q[c].mean(), q[c].min(), q[c].max(), q[ring].mean() if ring.any() else 1, (q[ring].mean() if ring.any() else 1) - q[c].mean(), t,
                 float(ys.min() == 0 or xs.min() == 0 or ys.max() == 127 or xs.max() == 127), a / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)), q.mean(), np.quantile(q, 0.05)]
            out.append((c, f))
    return out
def target(c, gt):
    hl, hk = ndi.label(~gt); return max([(c & (hl == j)).sum() / (c | (hl == j)).sum() for j in range(1, hk + 1)] + [0.0])
def select(C, pred, tau, kmax):
    cur = np.ones((128, 128), bool); used = np.zeros((128, 128), bool); n = 0
    for j in np.argsort(-pred):
        if pred[j] < tau or n >= kmax: break
        if (C[j][0] & used).any(): continue
        used |= C[j][0]; cur &= ~C[j][0]; n += 1
    return cur
tot_b, tot_n = [], []
for m, tag in (("pixelseal", "ens3_pixelseal"), ("trustmark", "ens2_trustmark"), ("stegastamp", "ens2_stegastamp")):
    names, P = load_maps(tag, "train"); sel = [i for i, n in enumerate(names) if df.loc[n].full_watermark]
    G = np.stack([g[names[i]] for i in sel]).astype(bool); Q = P[sel]; fold = df.loc[[names[i] for i in sel]].fold.values
    C = [cands(q) for q in Q]; Y = [np.array([target(c, gt) for c, _ in cc]) for cc, gt in zip(C, G)]
    base = np.zeros(len(Q)); new = np.zeros(len(Q))
    for k in range(5):
        tr = np.nonzero(fold != k)[0]; te = np.nonzero(fold == k)[0]
        t0 = max(TS, key=lambda t: miou_batch(Q[tr] > t, G[tr]).mean()); base[te] = miou_batch(Q[te] > t0, G[te])
        X = np.array([f for i in tr for _, f in C[i]]); y = np.concatenate([Y[i] for i in tr])
        torch.manual_seed(0); mu, sd = X.mean(0), X.std(0) + 1e-6; Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)); opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-3)
        for _ in range(400):
            opt.zero_grad(); ((net(Xt)[:, 0] - yt) ** 2).mean().backward(); opt.step()
        class GB:
            def predict(self, A): return net(torch.tensor((A - mu) / sd, dtype=torch.float32))[:, 0].detach().numpy()
        gb = GB()
        # inner predictions on train are optimistic; choose tau/kmax with them anyway (cheap), report held-out only
        pr = {i: gb.predict(np.array([f for _, f in C[i]])) if C[i] else np.zeros(0) for i in list(tr) + list(te)}
        best = max(((tau, km) for tau in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5) for km in (1, 2, 3, 5)),
                   key=lambda p: miou_batch(np.stack([select(C[i], pr[i], *p) for i in tr]), G[tr]).mean())
        new[te] = miou_batch(np.stack([select(C[i], pr[i], *best) for i in te]), G[te])
    tot_b.append(base.mean()); tot_n.append(new.mean())
    print(f"{m:11s} full n={len(Q)}  threshold CV {base.mean():.4f} | component rescoring CV {new.mean():.4f}  (delta {new.mean() - base.mean():+.4f}; paired share improved {np.mean(new > base + 1e-9):.2f}, worse {np.mean(new < base - 1e-9):.2f})", flush=True)
print(f"mean delta {np.mean(tot_n) - np.mean(tot_b):+.4f}")
