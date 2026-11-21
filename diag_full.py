"""Step 0 (full images): where are points lost? Uses the held-out maps of the current submission and rebuilds the
exact CV masks of postproc.py (gate + per-regime (sigma, op, threshold) fit on the training folds).
    python diag_full.py
Reports per method x true hole size: score, missed / bad-shape / fine groups, and oracles A (perfect location: keep only
the predicted hole component overlapping the truth best), B (our location, perfect shape: true hole shape centred on our
most confident hole component), C (best threshold per image, same smoothing/op)."""
import numpy as np, pandas as pd
from scipy import ndimage as ndi
import data, postproc
from diagnostics import best_gate, miou_batch
from evalmaps import gate_scores, load_maps

TAGS = {"pixelseal": "ens3_pixelseal", "trustmark": "ens2_trustmark", "stegastamp": "ens2_stegastamp"}


def cv_masks(tag, method):
    """Held-out masks + chosen params per image, exactly as postproc.evaluate."""
    df = data.train_table().set_index("name"); g = data.train_masks128()
    names, P = load_maps(tag, "train"); names = np.array(names); gs = gate_scores(method, "train")
    fold = df.loc[names].fold.values; full = df.loc[names].full_watermark.values
    G = np.stack([g[n] for n in names]); gsv = np.array([gs[n] for n in names]); S = postproc.table(P, G)
    M = np.zeros_like(G, bool); prm = [None] * len(P)
    for k in range(5):
        tr = fold != k; te = ~tr; pf = gsv > best_gate(gsv[tr], full[tr])
        for reg, regp in ((full, pf), (~full, ~pf)):
            m = S[..., tr & reg].mean(-1); s, o, t = np.unravel_index(np.argmax(m), m.shape)
            for i in np.nonzero(te & regp)[0]:
                p = ndi.gaussian_filter(P[i], postproc.SIGMAS[s]) if postproc.SIGMAS[s] else P[i]
                M[i] = postproc.op(p > postproc.TS[t], postproc.OPS[o]); prm[i] = (postproc.SIGMAS[s], postproc.OPS[o], postproc.TS[t])
    return names, P, G, M, prm, full, gsv


def score(m, g):
    return float(miou_batch(m[None], g[None])[0])


def main():
    rows = []
    for method, tag in TAGS.items():
        names, P, G, M, prm, full, gsv = cv_masks(tag, method)
        for i in np.nonzero(full)[0]:
            g, m, p = G[i].astype(bool), M[i], P[i]
            hole, phole = ~g, ~m
            nh = ndi.label(hole)[1]; frac = hole.mean()
            s = score(m, g)
            inter = (hole & phole).sum(); iou0 = inter / max((hole | phole).sum(), 1)
            grp = "no hole" if nh == 0 else ("missed" if inter == 0 else ("bad shape" if iou0 < 0.5 else "fine"))
            sg, o, t = prm[i]; ps = ndi.gaussian_filter(p, sg) if sg else p
            # oracle A: keep only the predicted hole component with the best overlap with the true hole
            lab, n = ndi.label(phole)
            if n and nh:
                best = max(range(1, n + 1), key=lambda k: (hole & (lab == k)).sum())
                a = score(~(lab == best), g) if (hole & (lab == best)).any() else s
            else:
                a = s
            # oracle B: true hole shape centred on our most confident hole component (lowest mean prob), else the map minimum
            if nh:
                if n:
                    k = min(range(1, n + 1), key=lambda k: ps[lab == k].mean()); cy, cx = ndi.center_of_mass(lab == k)
                else:
                    cy, cx = np.unravel_index(np.argmin(ps), ps.shape)
                hy, hx = ndi.center_of_mass(hole); sh = ndi.shift(hole.astype(float), (cy - hy, cx - hx), order=0) > 0.5
                b = score(~sh, g)
            else:
                b = s
            # oracle C: best threshold for this image (same smoothing and op)
            c = max(score(postproc.op(ps > tt, o), g) for tt in postproc.TS)
            rows.append(dict(method=method, size="none" if nh == 0 else ("small <3%" if frac < 0.03 else ("medium 3-10%" if frac < 0.10 else "large >10%")),
                             n_holes=nh, group=grp, score=s, A=max(a, s), B=b, C=max(c, s), frac=frac))
    R = pd.DataFrame(rows)
    print("== 1. holes per full train image")
    print(R.groupby("method").apply(lambda d: pd.Series({"images": len(d), "no hole": (d.n_holes == 0).sum(), "1 hole": (d.n_holes == 1).sum(),
                                                           "2+ holes": (d.n_holes >= 2).sum(), "median hole %": round(100 * d.frac.median(), 1)})).to_string())
    print("\n== 2. groups (count / mean score)")
    print(R.pivot_table(index="method", columns="group", values="score", aggfunc=["count", "mean"]).round(3).to_string())
    print("\n== by size: count / score / oracle A / B / C")
    print(R.groupby(["method", "size"]).agg(n=("score", "size"), score=("score", "mean"), A=("A", "mean"), B=("B", "mean"), C=("C", "mean")).round(3).to_string())
    print("\n== 3. totals (full images): current vs oracles, gain")
    T = R.groupby("method")[["score", "A", "B", "C"]].mean()
    for c in "ABC":
        T[f"gain {c}"] = T[c] - T["score"]
    print(T.round(3).to_string())
    print("\nall methods:", R[["score", "A", "B", "C"]].mean().round(3).to_dict())
    R.to_csv("results/diag_full.csv", index=False)


if __name__ == "__main__":
    main()
