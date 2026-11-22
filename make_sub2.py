"""Submission = per method: map (+ post-processing, postproc.fit_all) and, optionally, brute-force boxes
(bruterect.py) for images the gate calls partial whose box score > tau (tau fit on all train, like recteval.py).
    python make_sub2.py --name X --pixelseal TAG[:box[=boxtag]] ...   (boxtag: results/<boxtag>_<method>_{train,test}.npz, default brute)"""
import argparse
import numpy as np
import data, metrics, postproc
from diagnostics import miou_batch
from evalmaps import gate_scores, load_maps
from recteval import box_mask

KEY = "score"


def method_preds(m, spec):
    tag = spec.split(":")[0]; boxspec = spec.split(":")[1] if ":" in spec else ""
    use_box = boxspec.startswith("box"); btag = boxspec.split("=")[1] if "=" in boxspec else "brute"
    df = data.train_table().set_index("name"); g = data.train_masks128()
    names, P = load_maps(tag, "train"); G = np.stack([g[n] for n in names]); gs = gate_scores(m, "train")
    gsv = np.array([gs[n] for n in names]); full = df.loc[names].full_watermark.values
    prm = postproc.fit_all(P, G, gsv, full); Mtr = postproc.apply_all(P, gsv, prm)
    tn, Pt = load_maps(tag, "test"); gt = gate_scores(m, "test"); gtv = np.array([gt[n] for n in tn])
    Mte = postproc.apply_all(Pt, gtv, prm)
    print(m, tag, "post params", prm, "train fit", round(float(miou_batch(Mtr, G).mean()), 4))
    if use_box:
        pf = gsv > prm["gate"]
        z = np.load(data.RESULTS / f"{btag}_{m}_train.npz"); bx = dict(zip(z["names"].tolist(), z["box"])); bv = dict(zip(z["names"].tolist(), z[KEY]))
        has = np.array([n in bx for n in names]); val = np.array([bv.get(n, -1e9) for n in names])
        B = np.stack([box_mask(bx[n]) if n in bx else Mtr[i] for i, n in enumerate(names)])
        base = miou_batch(Mtr, G); boxs = miou_batch(B, G)
        cand = np.unique(np.quantile(val[has], np.linspace(0, 1, 41))).tolist() + [np.inf]
        sc = [np.where(~pf & has & (val > c), boxs, base).mean() for c in cand]; tau = cand[int(np.argmax(sc))]
        zt = np.load(data.RESULTS / f"{btag}_{m}_test.npz"); tbx = dict(zip(zt["names"].tolist(), zt["box"])); tbv = dict(zip(zt["names"].tolist(), zt[KEY]))
        n_used = 0
        for i, n in enumerate(tn):
            if gtv[i] <= prm["gate"] and n in tbx and tbv[n] > tau:
                Mte[i] = box_mask(tbx[n]); n_used += 1
        print(f"   boxes: tau {tau:.3f}, train fit {max(sc):.4f}, used on {n_used} test images")
    return dict(zip(tn, Mte.astype(np.uint8)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
    for m in data.METHODS:
        ap.add_argument(f"--{m}", required=True)
    a = ap.parse_args(); preds = {}
    for m in data.METHODS:
        preds.update(method_preds(m, getattr(a, m)))
    print("->", metrics.write_submission(preds, data.RESULTS / f"sub_{a.name}.pt"))


if __name__ == "__main__":
    main()
