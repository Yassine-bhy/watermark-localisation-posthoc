"""Kill test (CPU, ~2 min): per-image threshold chosen by EXPECTED mIoU under calibrated probabilities (full images).
The oracle per-image threshold is worth +0.038; a global threshold ignores how much hole mass an image has.
For each image and candidate threshold t (hole set H_t = {q < t}, q = calibrated P(watermarked)):
   E[IoU_hole] ~ sum_{c in H_t} (1 - q_c) / (|H_t| + sum_{c not in H_t} (1 - q_c))
   E[IoU_wm]   ~ sum_{c not in H_t} q_c   / (|not H_t| + sum_{c in H_t} q_c)
pick t maximising the mean (an empty hole set scores (0 + E[IoU_wm]) / 2, as in the official metric when a hole exists).
Calibration q = sigmoid(a * logit(p_smoothed) + b), (a, b) fit on the training folds' cells; smoothing sigma chosen on
the training folds.  Baseline: one global threshold fit on the training folds (same smoothing choice allowed).
Pass threshold (set in advance): >= +0.010 mean over the three methods and no method below baseline - 0.003."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch, data
from scipy import ndimage as ndi
from diagnostics import miou_batch
from evalmaps import load_maps
torch.set_num_threads(8)
df = data.train_table().set_index("name"); g = data.train_masks128(); TS = np.linspace(0.04, 0.96, 24); SIG = (0, 1.5, 3)
def calib(P, G):
    x = torch.tensor(np.log(np.clip(P, 1e-4, 1 - 1e-4) / (1 - np.clip(P, 1e-4, 1 - 1e-4))).ravel()[::7], dtype=torch.float32); y = torch.tensor(G.ravel()[::7].astype(np.float32))
    a = torch.ones(1, requires_grad=True); b = torch.zeros(1, requires_grad=True); opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=60)
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(a * x + b, y); l.backward(); return l
    opt.step(cl); return float(a), float(b)
def exp_mask(q):
    best = (-1, None)
    for t in np.linspace(0.02, 0.98, 49):
        H = q < t; nh = H.sum()
        if nh == 0: e = 0.5 * (q.sum() / q.size)
        else:
            e0 = (1 - q[H]).sum() / (nh + (1 - q[~H]).sum()); e1 = q[~H].sum() / ((~H).sum() + q[H].sum()) if (~H).any() else 0.0; e = 0.5 * (e0 + e1)
        if e > best[0]: best = (e, ~H)
    return best[1]
db, dn = [], []
for m, tag in (("pixelseal", "ens3_pixelseal"), ("trustmark", "ens2_trustmark"), ("stegastamp", "ens2_stegastamp")):
    names, P = load_maps(tag, "train"); sel = [i for i, n in enumerate(names) if df.loc[n].full_watermark]
    G = np.stack([g[names[i]] for i in sel]).astype(bool); fold = df.loc[[names[i] for i in sel]].fold.values
    Ps = {s: np.stack([ndi.gaussian_filter(p, s) if s else p for p in P[sel]]) for s in SIG}
    base = np.zeros(len(sel)); new = np.zeros(len(sel)); ch = []
    for k in range(5):
        tr = fold != k; te = ~tr
        s0, t0 = max(((s, t) for s in SIG for t in TS), key=lambda p: miou_batch(Ps[p[0]][tr] > p[1], G[tr]).mean()); base[te] = miou_batch(Ps[s0][te] > t0, G[te])
        cand = {}
        for s in SIG:
            a, b = calib(Ps[s][tr], G[tr]); f = lambda X: 1 / (1 + np.exp(-(a * np.log(np.clip(X, 1e-4, 1 - 1e-4) / (1 - np.clip(X, 1e-4, 1 - 1e-4))) + b)))
            cand[s] = (miou_batch(np.stack([exp_mask(f(x)) for x in Ps[s][tr]]), G[tr]).mean(), f)
        s1 = max(cand, key=lambda s: cand[s][0]); new[te] = miou_batch(np.stack([exp_mask(cand[s1][1](x)) for x in Ps[s1][te]]), G[te]); ch.append((s0, round(float(t0), 2), s1))
    db.append(base.mean()); dn.append(new.mean())
    print(f"{m:11s} full n={len(sel)}  global threshold CV {base.mean():.4f} | expected-mIoU threshold CV {new.mean():.4f}  (delta {new.mean() - base.mean():+.4f}; improved {np.mean(new > base + 1e-9):.2f} / worse {np.mean(new < base - 1e-9):.2f})  choices {ch}", flush=True)
print(f"mean delta {np.mean(dn) - np.mean(db):+.4f}")
