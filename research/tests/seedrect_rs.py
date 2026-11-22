"""Decoder-verified rectangle search seeded by RS-partial maps (for PixelSeal / StegaStamp rectangles).
    python research/tests/seedrect_rs.py <method> <train|test> <map_tag[,map_tag...]> <out_tag> [step=4] [radius=12]
Seeds per image: exact max-sum rectangles of (map - tau), tau = 0.3 and 0.5, plus the bounding boxes of the two
largest blobs of (map > 0.5).  Around every seed: translations -radius..radius in `step` px, height and width scaled
by 0.85 / 1.0 / 1.15 independently; the 5 best (known-message logit score, bruterect.Scorer) are refined with
bruterect.refine down to 1 px.  Only images whose whole-image decode says "not full" (global_soft <= 0.6) are searched.
Writes research/out/<out_tag>_<method>_<split>.npz in the bruterect format (names, box, score, agree): copy it to
results/<out_tag>_<method>_<split>.npz and use BRUTE_TAG=<out_tag> with recteval.py / make_sub2.py.
On the train split it also prints: how good the boxes are on true rectangles, and the CV value of the boxes with the
recteval.py rule (box replaces the post-processed map on gate-partial images when its score > tau, tau fit on the
training folds), for the map given as 7th argument (default: the first map tag).
Kill test (written before running): StegaStamp boxes with IoU > 0.8 on >= 40 % of the true rectangles (the old
searches reach 5 %), or a CV gain >= +0.01 over the same map without boxes."""
import sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F
from scipy import ndimage as ndi
import data, postproc
from bruterect import Scorer, ok, refine, iou
from diagnostics import best_gate, miou_batch
from evalmaps import gate_scores, load_maps

method, split, tags, out_tag = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
STEP = int(sys.argv[5]) if len(sys.argv) > 5 else 4; RAD = int(sys.argv[6]) if len(sys.argv) > 6 else 12
CVMAP = sys.argv[7] if len(sys.argv) > 7 else tags[0]; t0 = time.time(); LIM = 2.2 if method == "trustmark" else 16.5


def load_any(tag, sp):
    try:
        z = np.load(f"research/out/{tag}_{sp}.npz"); return dict(zip(z["names"].tolist(), z["prob"].astype(np.float32)))
    except FileNotFoundError:
        n, P = load_maps(tag, sp); return dict(zip(n, P))


def maxrect(p, tau):
    q = F.avg_pool2d(torch.from_numpy(p - tau)[None, None], 2)[0, 0] * 4; n = q.shape[0]
    c = F.pad(q.cumsum(0).cumsum(1), (1, 0, 1, 0)); best = (-1e9, None)
    for h in range(1, n + 1):
        A = c[h:, :] - c[:-h, :]
        for w in range(max(1, int(np.ceil(h / LIM))), min(n, int(h * LIM)) + 1):
            S = A[:, w:] - A[:, :-w]; v, k = S.flatten().max(0)
            if v > best[0]:
                y, x = divmod(int(k), S.shape[1]); best = (float(v), (y * 8, x * 8, (y + h) * 8, (x + w) * 8))
    return best[1]


def seeds(p):
    out = [maxrect(p, 0.3), maxrect(p, 0.5)]
    lab, n = ndi.label(p > 0.5)
    if n:
        sz = ndi.sum(p, lab, range(1, n + 1))
        for j in np.argsort(sz)[::-1][:2]:
            ys, xs = np.nonzero(lab == j + 1)
            if len(ys) >= 4:
                out.append((ys.min() * 4, xs.min() * 4, (ys.max() + 1) * 4, (xs.max() + 1) * 4))
    keep = []
    for b in out:
        if b is not None and all(iou(b, k) < 0.7 for k in keep):
            keep.append(b)
    return keep[:4]


maps = [load_any(t, split) for t in tags]; sc = Scorer(method)
gs = dict(zip(*[np.load(data.RESULTS / "signals" / f"{method}_{split}.npz")[k].tolist() for k in ("names", "global_soft")]))
ds = data.load_split(split, [method]); out = {"names": [], "box": [], "score": [], "agree": []}; nb = 0
for i, m in enumerate(ds.meta):
    n = m["image_name"]
    if gs[n] > 0.6:
        continue
    it = ds[i]; x = torch.from_numpy(it["image"].copy()).permute(2, 0, 1)[None].float().div(255).cuda()
    s = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).cuda(); p = np.mean([mp[n] for mp in maps], 0); cands = set()
    for y0, x0, y1, x1 in seeds(p):
        cy, cx, h, w = (y0 + y1) / 2, (x0 + x1) / 2, y1 - y0, x1 - x0
        for fh in (0.85, 1.0, 1.15):
            for fw in (0.85, 1.0, 1.15):
                for dy in range(-RAD, RAD + 1, STEP):
                    for dx in range(-RAD, RAD + 1, STEP):
                        b = (int(cy + dy - h * fh / 2), int(cx + dx - w * fw / 2), int(cy + dy + h * fh / 2), int(cx + dx + w * fw / 2))
                        if ok(b, LIM):
                            cands.add(b)
    cands = list(cands); nb += len(cands)
    if not cands:
        best = ((0, 0, 512, 512), -1e9, -1.0)
    else:
        v, _ = sc.score(x, cands, s); top = []
        for j in np.argsort(-v):
            if all(iou(cands[j], cands[t]) < 0.5 for t in top):
                top.append(j)
            if len(top) == 5:
                break
        bb = max((refine(sc, x, s, cands[j], float(v[j]), LIM) for j in top), key=lambda r: r[1])
        _, ag = sc.score(x, [bb[0]], s); best = (bb[0], bb[1], float(ag[0]))
    for k, val in zip(("names", "box", "score", "agree"), (n, best[0], best[1], best[2])):
        out[k].append(val)
    if len(out["names"]) % 20 == 0:
        print(len(out["names"]), f"{time.time() - t0:.0f}s  ({nb / len(out['names']):.0f} boxes per image)", flush=True)
try:
    np.savez(f"research/out/{out_tag}_{method}_{split}.npz", **{k: np.array(v) for k, v in out.items()})
    print(f"wrote research/out/{out_tag}_{method}_{split}.npz  ({len(out['names'])} images)  {time.time() - t0:.0f}s", flush=True)
except OSError as e:
    print("boxes not saved:", e, flush=True)

if split == "train":
    df = data.train_table().set_index("name"); g = data.train_masks128(); bx = dict(zip(out["names"], out["box"])); bv = dict(zip(out["names"], out["score"]))
    # 1. box quality on true rectangles
    q = []
    for i, m in enumerate(ds.meta):
        n = m["image_name"]
        if df.loc[n].full_watermark or n not in bx:
            continue
        gm = ds[i]["mask"].astype(bool); ys, xs = np.nonzero(gm)
        if not len(ys) or gm[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean() < 0.95:
            continue
        q.append((iou(tuple(int(v) for v in bx[n]), (ys.min(), xs.min(), ys.max() + 1, xs.max() + 1)), bv[n]))
    q = np.array(q)
    print(f"true rectangles with a box: {len(q)} | box IoU median {np.median(q[:, 0]):.2f}, > 0.8: {np.mean(q[:, 0] > 0.8):.2f}, > 0.5: {np.mean(q[:, 0] > 0.5):.2f} | "
          f"score of good (IoU > 0.8) vs other boxes, median: {np.median(q[q[:, 0] > 0.8, 1]) if (q[:, 0] > 0.8).any() else float('nan'):.2f} vs {np.median(q[q[:, 0] <= 0.8, 1]) if (q[:, 0] <= 0.8).any() else float('nan'):.2f}")
    # 2. CV value with the recteval.py rule
    mp_ = load_any(CVMAP, "train"); names = np.array([n for n in df.index if df.loc[n].method == method]); P = np.stack([mp_[n] for n in names])
    gsc = gate_scores(method, "train"); gsv = np.array([gsc[n] for n in names]); fold = df.loc[names].fold.values; full = df.loc[names].full_watermark.values
    G = np.stack([g[n] for n in names]); S = postproc.table(P, G)

    def box_mask(b):
        mk = np.zeros((128, 128), bool); y0, x0, y1, x1 = [int(round(v / 4)) for v in b]; mk[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = True; return mk
    has = np.array([n in bx for n in names]); val = np.array([bv.get(n, -1e9) for n in names])
    Bs = np.array([miou_batch(box_mask(bx[n])[None], G[i][None])[0] if has[i] else 0 for i, n in enumerate(names)]); hm = np.zeros(len(P)); hb = np.zeros(len(P))
    for k in range(5):
        tr = fold != k; te = ~tr; pf = gsv > best_gate(gsv[tr], full[tr]); mp2 = np.zeros(len(P))
        for reg, regp in ((full, pf), (~full, ~pf)):
            mm = S[..., tr & reg].mean(-1); s_, o_, t_ = np.unravel_index(np.argmax(mm), mm.shape); mp2[regp] = S[s_, o_, t_, regp]
        cand = np.unique(np.quantile(val[has], np.linspace(0, 1, 41))).tolist() + [np.inf]
        c = cand[int(np.argmax([np.where(~pf & has & (val > c), Bs, mp2)[tr].mean() for c in cand]))]
        hm[te] = mp2[te]; hb[te] = np.where(~pf & has & (val > c), Bs, mp2)[te]
    print(f"CV with the recteval rule on map {CVMAP}: all {hm.mean():.4f} -> {hb.mean():.4f} | partial {hm[~full].mean():.4f} -> {hb[~full].mean():.4f}", flush=True)
print("done", f"{time.time() - t0:.0f}s")
