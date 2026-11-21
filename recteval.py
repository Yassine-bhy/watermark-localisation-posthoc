"""CV value of the brute-force rectangles (bruterect.py) on top of a map (+ post-processing of postproc.py).
    python recteval.py <map_tag> [score|agree]
Per held-out fold: gate, map post-processing params and the box cut-off tau are fit on the training folds;
an image the gate calls partial gets its box as mask if its box score > tau, else the post-processed map."""
import os
import sys
import numpy as np
import data
from diagnostics import best_gate, miou_batch
from evalmaps import gate_scores, load_maps
import postproc


def box_mask(b):
    m = np.zeros((128, 128), bool); y0, x0, y1, x1 = [int(round(v / 4)) for v in b]; m[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = True
    return m


def main(tag, key="agree"):
    method = [m for m in data.METHODS if m in tag][0]
    df = data.train_table().set_index("name"); g = data.train_masks128()
    names, P = load_maps(tag, "train"); names = np.array(names); gs = gate_scores(method, "train")
    fold = df.loc[names].fold.values; full = df.loc[names].full_watermark.values; regime = df.loc[names].regime.values
    G = np.stack([g[n] for n in names]); gsv = np.array([gs[n] for n in names])
    z = np.load(data.RESULTS / f"{os.environ.get('BRUTE_TAG', 'brute')}_{method}_train.npz"); bx = dict(zip(z["names"].tolist(), z["box"])); bv = dict(zip(z["names"].tolist(), z[key]))
    has = np.array([n in bx for n in names]); val = np.array([bv.get(n, -1e9) for n in names])
    B = np.array([miou_batch(box_mask(bx[n])[None], G[i][None])[0] if has[i] else 0 for i, n in enumerate(names)])
    S = postproc.table(P, G)
    held_map = np.zeros(len(P)); held = np.zeros(len(P))
    for k in range(5):
        tr = fold != k; te = ~tr; pf = gsv > best_gate(gsv[tr], full[tr]); mp = np.zeros(len(P))
        for reg, regp in ((full, pf), (~full, ~pf)):
            m = S[..., tr & reg].mean(-1); s, o, t = np.unravel_index(np.argmax(m), m.shape); mp[regp] = S[s, o, t, regp]
        cand = np.unique(np.quantile(val[has], np.linspace(0, 1, 41))).tolist() + [np.inf]
        sc = [np.where(~pf & has & (val > c), B, mp)[tr].mean() for c in cand]; c = cand[int(np.argmax(sc))]
        use = ~pf & has & (val > c)
        held[te] = np.where(use, B, mp)[te]; held_map[te] = mp[te]
        print(f"fold {k}: tau {c:.3f}  boxes used on held-out {use[te].sum()}", flush=True)
    main.result = (held_map.mean(), held.mean())
    print(f"{tag}: map+post CV {held_map.mean():.4f} -> with boxes {held.mean():.4f}  (partial {held_map[~full].mean():.4f} -> {held[~full].mean():.4f})")
    for r in ("partial",):
        pass
    sel = has & ~full
    print("box IoU score on partial images with a box: median", np.median(B[sel]).round(3), "| oracle (box when better):", np.maximum(B, held_map)[~full].mean().round(4))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or ["agree"]))
