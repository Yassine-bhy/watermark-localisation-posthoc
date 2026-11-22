"""Step 2 bis diagnosis: which shape errors dominate on full images with a found-but-bad hole (IoU_0 < 0.5)?
Classes (first that applies): 'missed some holes' (>= 1 true hole component untouched while another is found),
'merged' (one predicted hole component covers >= 2 true holes), 'too small' (pred hole area < 0.5 x true),
'too big / blurry' (> 2 x true), 'misplaced / wrong shape' (rest)."""
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from diag_full import TAGS, cv_masks, score

rows = []
for method, tag in TAGS.items():
    names, P, G, M, prm, full, gsv = cv_masks(tag, method)
    for i in np.nonzero(full)[0]:
        g = G[i].astype(bool); hole, ph = ~g, ~M[i]
        inter = (hole & ph).sum(); iou0 = inter / max((hole | ph).sum(), 1)
        if inter == 0 or iou0 >= 0.5:
            continue
        lt, nt = ndi.label(hole); lp, npred = ndi.label(ph)
        touched = [k for k in range(1, nt + 1) if (ph & (lt == k)).any()]
        merged = any(len(set(np.unique(lt[lp == k])) - {0}) >= 2 for k in range(1, npred + 1))
        ratio = ph.sum() / hole.sum()
        missed_area = sum((lt == k).sum() for k in range(1, nt + 1) if k not in touched) / hole.sum()
        c = ("missed some holes" if len(touched) < nt and missed_area > 0.25 else "merged" if merged else
             "too small" if ratio < 0.5 else "too big / blurry" if ratio > 2 else "misplaced / wrong shape")
        rows.append(dict(method=method, cls=c, score=score(M[i], g), ratio=ratio, n_true=nt, n_pred=npred))
R = pd.DataFrame(rows)
print(R.pivot_table(index="cls", columns="method", values="score", aggfunc="count", fill_value=0).to_string())
print("\nmean score per class:", R.groupby("cls").score.mean().round(3).to_dict())
print("loss (sum of 1 - score) per class:", R.groupby("cls").score.apply(lambda s: round(float((1 - s).sum()), 1)).to_dict())
print("pred/true hole area ratio: median", round(float(R.ratio.median()), 2), "| quartiles", np.round(R.ratio.quantile([.25, .75]).values, 2))
print("predicted hole components: median", R.n_pred.median(), " true:", R.n_true.median())
