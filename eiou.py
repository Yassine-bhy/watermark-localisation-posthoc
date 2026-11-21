"""Idea 4: per image, the mask (top-k cells of the map) that maximises the EXPECTED official score
(IoU_0 + IoU_1)/2 under calibrated cell probabilities q (instead of one global threshold).
    python eiou.py <tag> ...
Calibration: q = sigmoid(T * (logit(p) - c)), after Gaussian smoothing sigma; (sigma, T, c) chosen per regime on the
training folds (CV, gate as in evalmaps)."""
import itertools, sys
import numpy as np, pandas as pd
from scipy import ndimage as ndi
import data
from diagnostics import best_gate, miou_batch
from evalmaps import gate_scores, load_maps

GRID = list(itertools.product((0, 3, 6), (0.5, 1, 2), (-1.5, -0.5, 0.5, 1.5)))


def best_mask(q):
    """q: 128 x 128 probabilities -> binary mask maximising expected mIoU over top-k sets (k = 0..N)."""
    f = q.ravel(); o = np.argsort(-f); c = np.concatenate([[0], np.cumsum(f[o])]); N = f.size; S = c[-1]; k = np.arange(N + 1)
    iou1 = np.where(k + S - c > 0, c / np.maximum(k + S - c, 1e-9), 1.0)
    iou0 = np.where(N - c > 0, ((N - k) - (S - c)) / np.maximum(N - c, 1e-9), 1.0)
    kb = int(np.argmax(iou1 + iou0)); m = np.zeros(N, bool); m[o[:kb]] = True
    return m.reshape(q.shape)


def masks(P, sg, T, c):
    L = np.log(np.clip(P, 1e-4, 1 - 1e-4) / (1 - np.clip(P, 1e-4, 1 - 1e-4)))
    out = []
    for l in L:
        l = ndi.gaussian_filter(l, sg) if sg else l
        out.append(best_mask(1 / (1 + np.exp(-T * (l - c)))))
    return np.stack(out)


def evaluate(tag):
    method = [m for m in data.METHODS if m in tag][0]
    df = data.train_table().set_index("name"); g = data.train_masks128()
    names, P = load_maps(tag, "train"); gs = gate_scores(method, "train"); names = np.array(names)
    fold = df.loc[names].fold.values; full = df.loc[names].full_watermark.values
    G = np.stack([g[n] for n in names]); gsv = np.array([gs[n] for n in names])
    S = np.stack([miou_batch(masks(P, *prm), G) for prm in GRID])        # params x images
    held = np.zeros(len(P))
    for k in range(5):
        tr = fold != k; te = ~tr; pf = gsv > best_gate(gsv[tr], full[tr])
        for reg, regp in ((full, pf), (~full, ~pf)):
            j = int(np.argmax(S[:, tr & reg].mean(1))); held[te & regp] = S[j, te & regp]
    return dict(tag=tag, cv=held.mean(), cv_full=held[full].mean(), cv_part=held[~full].mean(),
                best_full=GRID[int(np.argmax(S[:, full].mean(1)))], best_part=GRID[int(np.argmax(S[:, ~full].mean(1)))])


if __name__ == "__main__":
    print(pd.DataFrame([evaluate(t) for t in sys.argv[1:]]).round(4).to_string(index=False))
