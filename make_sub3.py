"""Submission with the full partial pipeline of pipecv.py, decisions fit on ALL train images:
gate -> full: post-processed map | partial: box if score > tau_box, else SAM pick if agree >= tau_sam, else map.
    python make_sub3.py --name X --pixelseal MAP[:BOX][:sam] --trustmark ... --stegastamp ...
e.g. --trustmark ens2_trustmark:brute:sam  (BOX = results/<BOX>_<m>_{train,test}.npz; omit for no boxes)."""
import argparse
import numpy as np
import data, metrics, postproc
from diag_full import score
from evalmaps import gate_scores, load_maps
from pipecv import maxrect, sam_pick, zoom_feats
from recteval import box_mask


def method_preds(m, spec):
    parts = spec.split(":"); tag = parts[0]; use_sam = "sam" in parts[1:]
    zoom = next((p[5:] for p in parts[1:] if p.startswith("zoom=")), "")
    box = next((p for p in parts[1:] if p != "sam" and not p.startswith("zoom=")), "")
    df = data.train_table().set_index("name"); g = data.train_masks128()
    names, P = load_maps(tag, "train"); G = np.stack([g[n] for n in names]); gs = gate_scores(m, "train")
    gsv = np.array([gs[n] for n in names]); full = df.loc[names].full_watermark.values
    prm = postproc.fit_all(P, G, gsv, full); Mtr = postproc.apply_all(P, gsv, prm); part = gsv <= prm["gate"]
    base = np.array([score(Mtr[i], G[i]) for i in range(len(P))])

    def extras(split, nm, Pm, Mm):
        bs = np.full(len(nm), -np.inf); bm = [None] * len(nm); sag = np.full(len(nm), -1.0); sm = [None] * len(nm)
        if box:
            z = np.load(data.RESULTS / f"{box}_{m}_{split}.npz"); bx = dict(zip(z["names"].tolist(), z["box"])); bv = dict(zip(z["names"].tolist(), z["score"]))
            for i, n in enumerate(nm):
                if n in bx:
                    bs[i] = bv[n]; bm[i] = box_mask(bx[n])
        if use_sam:
            z = np.load(data.RESULTS / f"samcache_{m}_{split}.npz"); cm = dict(zip(z["names"].tolist(), z["masks"])); cp = dict(zip(z["names"].tolist(), z["point"]))
            for i, n in enumerate(nm):
                if n in cm:
                    pick, ag = sam_pick(list(cm[n]), Pm[i], Mm[i], tuple(cp[n]))
                    if pick is not None:
                        sag[i] = ag; sm[i] = pick
        zr = np.full(len(nm), -np.inf); zm = [None] * len(nm)
        if zoom:
            zn, Zm = load_maps(zoom, split); Zd = dict(zip(zn, Zm)); Z = np.stack([Zd[n] for n in nm])
            zr = zoom_feats(Z, Pm); zm = [maxrect(Z[i], 0.3) for i in range(len(nm))]
        return bs, bm, sag, sm, zr, zm
    bs, bm, sag, sm, zr, zm = extras("train", names, P, Mtr)
    zsc = np.array([score(zm[i], G[i]) if zm[i] is not None else 0 for i in range(len(P))])
    bsc = np.array([score(bm[i], G[i]) if bm[i] is not None else 0 for i in range(len(P))])
    ssc = np.array([score(sm[i], G[i]) if sm[i] is not None else 0 for i in range(len(P))])
    TB = [np.inf] + (list(np.unique(np.quantile(bs[np.isfinite(bs)], np.linspace(0, 1, 41)))) if box else [])
    TA = [np.inf] + ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] if use_sam else [])
    TZ = [np.inf] + (list(np.quantile(zr[part], np.linspace(0, 0.95, 20))) if zoom else [])

    def pipe(tb, tz, ta):
        ub = part & (bs > tb); uz = part & ~ub & (zr > tz); us = part & ~ub & ~uz & (sag >= ta)
        return np.where(ub, bsc, np.where(uz, zsc, np.where(us, ssc, base))).mean()
    best = max(((pipe(tb, tz, ta), tb, tz, ta) for tb in TB for tz in TZ for ta in TA), key=lambda r: r[0])
    _, tb, tz, ta = best
    tn, Pt = load_maps(tag, "test"); gt = gate_scores(m, "test"); gtv = np.array([gt[n] for n in tn])
    Mte = postproc.apply_all(Pt, gtv, prm); pt = gtv <= prm["gate"]
    tbs, tbm, tsag, tsm, tzr, tzm = extras("test", tn, Pt, Mte); nb = nz = ns = 0
    for i in range(len(tn)):
        if pt[i] and tbs[i] > tb:
            Mte[i] = tbm[i]; nb += 1
        elif pt[i] and tzr[i] > tz:
            Mte[i] = tzm[i]; nz += 1
        elif pt[i] and tsag[i] >= ta and tsm[i] is not None:
            Mte[i] = tsm[i]; ns += 1
    print(f"{m} {spec}: train fit {best[0]:.4f} (map only {base.mean():.4f}); tau_box {tb:.2f}, rho_zoom {tz:.3f}, tau_sam {ta}; test: {nb} boxes, {nz} zoom rects, {ns} SAM masks", flush=True)
    return dict(zip(tn, Mte.astype(np.uint8)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
    for m in data.METHODS:
        ap.add_argument(f"--{m}", required=True)
    a = ap.parse_args(); preds = {}
    for m in data.METHODS:
        preds.update(method_preds(m, getattr(a, m)))
    import time
    for t in range(20):                     # intermittent project inode quota: retry the save
        try:
            print("->", metrics.write_submission(preds, data.RESULTS / f"sub_{a.name}.pt")); break
        except (RuntimeError, OSError) as e:
            print("save failed, retry", t, e, flush=True); time.sleep(20)
