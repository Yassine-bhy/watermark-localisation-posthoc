"""Step 1 (full images): cheap post-processing rules on top of the current CV masks (diag_full.cv_masks), applied only to
images the gate calls full; rule parameters chosen on the training folds; scored on held-out full images.
Rules: never-empty (no predicted hole -> add one at the most confident location, size = median true hole size of the
method on the training folds); keep top-k hole components by summed confidence (1 - p); min hole component size."""
import numpy as np, pandas as pd
from scipy import ndimage as ndi
import data
from diag_full import TAGS, cv_masks, score


def never_empty(m, p, size):
    if (~m).any():
        return m
    o = np.argsort(ndi.gaussian_filter(p, 3).ravel())[:size]; m = m.copy().ravel(); m[o] = False
    return m.reshape(128, 128)


def topk(m, p, k, minsz):
    lab, n = ndi.label(~m)
    if n == 0:
        return m
    conf = [((1 - p)[lab == j]).sum() for j in range(1, n + 1)]; sz = [(lab == j).sum() for j in range(1, n + 1)]
    keep = [j + 1 for j in np.argsort(conf)[::-1][:k] if sz[j] >= minsz]
    return ~np.isin(lab, keep)


df = data.train_table().set_index("name"); out = []
for method, tag in TAGS.items():
    names, P, G, M, prm, full, gsv = cv_masks(tag, method); fold = df.loc[names].fold.values
    pf = np.array([pp is not None and True for pp in prm])                       # all images have masks
    # which images the CV gate called full: their params came from the full regime -> recompute gate per fold
    from diagnostics import best_gate
    gated_full = np.zeros(len(P), bool)
    for k in range(5):
        tr = fold != k; gated_full[~tr] = (gsv > best_gate(gsv[tr], full[tr]))[~tr]
    base = np.array([score(M[i], G[i]) for i in range(len(P))])
    variants = {"never-empty": [], **{f"top{k} min{s}": [] for k in (1, 2, 3, 5) for s in (0, 8, 32)}}
    res = {v: base.copy() for v in variants}
    for k in range(5):
        tr = fold != k; te = ~tr
        hole_med = int(np.median([(~G[i].astype(bool)).sum() for i in np.nonzero(tr & full)[0]]))
        for i in np.nonzero(te & gated_full)[0]:
            res["never-empty"][i] = score(never_empty(M[i], P[i], hole_med), G[i])
            for kk in (1, 2, 3, 5):
                for s in (0, 8, 32):
                    res[f"top{kk} min{s}"][i] = score(topk(M[i], P[i], kk, s), G[i])
    small = full & np.array([(~G[i].astype(bool)).mean() < 0.03 for i in range(len(P))])
    n_empty = sum(1 for i in np.nonzero(gated_full)[0] if not (~M[i]).any())
    row = {"method": method, "base full": base[full].mean(), "base small": base[small].mean(), "no-hole preds": n_empty}
    for v in variants:
        row[v + " full"] = res[v][full].mean(); row[v + " small"] = res[v][small].mean()
    out.append(row)
R = pd.DataFrame(out).set_index("method").T
print(R.round(4).to_string())
