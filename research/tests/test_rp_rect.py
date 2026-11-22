"""Turning RS-partial maps into masks on PARTIAL images: threshold, exact max-sum rectangle, or a routed choice.
    python research/tests/test_rp_rect.py <method> <map_tag[,map_tag...]> [old_tag]      (CPU, ~3 min)
Maps are averaged (tags from research/out or results/preds; held-out train maps). True regime = partial (kill test).
  thr     one global threshold (fit on the training folds)
  rect    exact max-sum rectangle of (map - tau) on the 64 grid, aspect within the method's limit
  routed  rectangle if it agrees with the largest thresholded blob (IoU >= rho), else the thresholded map
          (tau, rho, threshold fit on the training folds)
  oracle  rectangle on true rectangles, threshold on true objects (ceiling of any routing rule)
Pass threshold (written before running): 'routed' >= 'thr' + 0.03 on partial images, held-out."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F, data
from scipy import ndimage as ndi
from diagnostics import miou_batch
from evalmaps import load_maps
torch.set_num_threads(8)
method = sys.argv[1]; tags = sys.argv[2].split(","); old = sys.argv[3] if len(sys.argv) > 3 else None
df = data.train_table().set_index("name"); g = data.train_masks128(); TS = np.linspace(0.04, 0.96, 24); LIM = 2.2 if method == "trustmark" else 16.5


def load_any(tag):
    try:
        z = np.load(f"research/out/{tag}_train.npz"); return dict(zip(z["names"].tolist(), z["prob"].astype(np.float32)))
    except FileNotFoundError:
        n, P = load_maps(tag, "train"); return dict(zip(n, P))


def maxrect(p, tau):
    q = F.avg_pool2d(torch.from_numpy(p - tau)[None, None], 2)[0, 0] * 4; n = q.shape[0]
    c = F.pad(q.cumsum(0).cumsum(1), (1, 0, 1, 0)); best = (-1e9, None)
    for h in range(1, n + 1):
        A = c[h:, :] - c[:-h, :]
        for w in range(max(1, int(np.ceil(h / LIM))), min(n, int(h * LIM)) + 1):
            S = A[:, w:] - A[:, :-w]; v, k = S.flatten().max(0)
            if v > best[0]:
                y, x = divmod(int(k), S.shape[1]); best = (float(v), (y, x, y + h, x + w))
    m = np.zeros((128, 128), bool)
    if best[1]:
        y0, x0, y1, x1 = [2 * t for t in best[1]]; m[y0:y1, x0:x1] = True
    return m


def blob(b):
    lab, k = ndi.label(b)
    if not k: return b
    sz = ndi.sum(b, lab, range(1, k + 1)); return lab == 1 + int(np.argmax(sz))


iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
M = [load_any(t) for t in tags]; names = [n for n in df.index if df.loc[n].method == method and not df.loc[n].full_watermark]
P = np.stack([np.mean([m[n] for m in M], 0) for n in names]); G = np.stack([g[n] for n in names]).astype(bool); fold = df.loc[names].fold.values


def isrect(m):
    ys, xs = np.nonzero(m); return len(ys) > 0 and m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean() > 0.95


rect = np.array([isrect(x) for x in G]); TAUS = (0.2, 0.3, 0.4, 0.5, 0.6); RHOS = (0.3, 0.4, 0.5, 0.6, 0.7)
R = {tau: np.stack([maxrect(p, tau) for p in P]) for tau in TAUS}
sc = lambda Mk: miou_batch(Mk, G)
res = {k: np.zeros(len(P)) for k in ("thr", "rect", "routed", "oracle")}
for k in range(5):
    tr = fold != k; te = ~tr
    t = max(TS, key=lambda t: sc(P > t)[tr].mean()); T = P > t; res["thr"][te] = sc(T)[te]
    tau = max(TAUS, key=lambda a: sc(R[a])[tr].mean()); res["rect"][te] = sc(R[tau])[te]
    tau_r = max(TAUS, key=lambda a: sc(R[a])[tr & rect].mean()); res["oracle"][te] = np.where(rect, sc(R[tau_r]), sc(T))[te]
    B = np.stack([blob(x) for x in T]); best = (-1, None)
    for a in TAUS:
        agree = np.array([iou(B[i], R[a][i]) for i in range(len(P))])
        for rho in RHOS:
            m_ = np.where((agree >= rho)[:, None, None], R[a], T); s = sc(m_)[tr].mean()
            if s > best[0]: best = (s, (a, rho))
    a, rho = best[1]; agree = np.array([iou(B[i], R[a][i]) for i in range(len(P))])
    res["routed"][te] = sc(np.where((agree >= rho)[:, None, None], R[a], T))[te]
print(f"{method}  maps = mean({', '.join(tags)})   partial images {len(P)} ({int(rect.sum())} rectangles, {int((~rect).sum())} objects)")
if old:
    Po = load_any(old); Po = np.stack([Po[n] for n in names]); o = np.zeros(len(P))
    for k in range(5):
        tr = fold != k; t = max(TS, key=lambda t: sc(Po > t)[tr].mean()); o[~tr] = sc(Po > t)[~tr]
    print(f"   {old + ' thr':22s} partial {o.mean():.4f} | rectangles {o[rect].mean():.3f} | objects {o[~rect].mean():.3f}")
for k, v in res.items():
    print(f"   {k:22s} partial {v.mean():.4f} | rectangles {v[rect].mean():.3f} | objects {v[~rect].mean():.3f}")
d = res["routed"] - res["thr"]; rng = np.random.default_rng(0); boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
print(f"   routed vs thr: {d.mean():+.4f}  [{np.quantile(boot, .025):+.4f}, {np.quantile(boot, .975):+.4f}]")
