"""Step 1 diagnostics: how well does each signal map separate watermarked from clean cells?
    python diagnostics.py            (needs results/signals/*_{train,test}.npz from signals.py)
Per method and signal: cell-level AUC over all train cells, and 5-fold CV mIoU of two simple predictors
  - 'thr'    : one threshold on the signal (chosen on the 4 training folds, scored on the held-out fold)
  - 'regime' : first full vs partial from the whole-image agreement (threshold chosen on training folds),
               then a separate threshold per regime (±inf allowed, i.e. a constant mask)
Signals are also tried per-image standardised ('_z': (s - median) / MAD within the image).
Writes results/step1_diag.csv, results/step1_regime.csv, results/step1_cv_<method>.npz (held-out
predictions of the best predictor) and the test submission results/sub_step1_signals.pt."""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import data
import metrics

QS = np.linspace(0.01, 0.99, 49)


def miou_batch(P, G):
    """Vectorised official mIoU per image (checked against wmloc.utils.miou_image in main)."""
    out = 0
    for c in (0, 1):
        p, g = P == c, G == c
        inter = (p & g).sum((1, 2)); union = (p | g).sum((1, 2))
        out = out + np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return out / 2


def auc(scores, labels):
    r = rankdata(scores); n1 = labels.sum(); n0 = labels.size - n1
    return (r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def zscore(S):
    med = np.median(S, (1, 2), keepdims=True)
    mad = np.median(np.abs(S - med), (1, 2), keepdims=True) + 1e-6
    return (S - med) / mad


def candidates(S):
    return np.concatenate([[-np.inf], np.quantile(S, QS), [np.inf]])


def best_thr(S, G):
    """Threshold t maximising mean mIoU of (S > t) on these images."""
    ts = candidates(S)
    sc = [miou_batch(S > t, G).mean() for t in ts]
    return ts[int(np.argmax(sc))]


def best_gate(g, full):
    """Threshold on the whole-image agreement that best separates full from partial (accuracy)."""
    ts = np.unique(g)
    acc = [((g > t) == full).mean() for t in ts]
    return ts[int(np.argmax(acc))]


def fit(S, G, gsoft, full, variant):
    if variant == "thr":
        return {"t": best_thr(S, G)}
    return {"gate": best_gate(gsoft, full), "t_full": best_thr(S[full], G[full]), "t_part": best_thr(S[~full], G[~full])}


def predict(S, gsoft, prm):
    if "t" in prm:
        return S > prm["t"]
    pf = gsoft > prm["gate"]
    return np.where(pf[:, None, None], S > prm["t_full"], S > prm["t_part"])


WITH_B = False   # --with-b: also load the step-1b maps from results/signals_b


def load(method, split):
    z = np.load(data.RESULTS / "signals" / f"{method}_{split}.npz")
    sig = {k: z[k].astype(np.float32) for k in z.files if k not in ("names", "global_soft", "global_acc")}
    if WITH_B:
        zb = np.load(data.RESULTS / "signals_b" / f"{method}_{split}.npz")
        assert (zb["names"] == z["names"]).all()
        sig.update({k: zb[k].astype(np.float32) for k in zb.files if k != "names"})
    sig["winmean"] = (sig["win64"] + sig["win128"] + sig["win256"]) / 3
    for k in list(sig):
        sig[k + "_z"] = zscore(sig[k])
    return z["names"].tolist(), sig, z["global_soft"], z["global_acc"]


def main():
    global WITH_B
    import sys
    WITH_B = "--with-b" in sys.argv
    tag = "step1b" if WITH_B else "step1"
    df = data.train_table(); gts = data.train_masks128()
    rows, regime_rows, test_preds = [], [], {}
    for method in data.METHODS:
        names, sig, gsoft, gacc = load(method, "train")
        t = df.set_index("name").loc[names]
        G = np.stack([gts[n] for n in names]).astype(np.uint8)
        full = t.full_watermark.values; fold = t.fold.values

        # regime classifier alone (CV accuracy)
        pf = np.zeros(len(names), bool)
        for k in range(5):
            tr = fold != k
            pf[~tr] = gsoft[~tr] > best_gate(gsoft[tr], full[tr])
        regime_rows.append(dict(method=method, cv_accuracy=(pf == full).mean(),
                                acc_full=(pf[full]).mean(), acc_partial=(~pf[~full]).mean(),
                                auc_global_soft=auc(gsoft, full)))

        best = None
        for name, S in sig.items():
            a = auc(S.ravel(), G.ravel().astype(bool))
            if a < 0.5:
                S = -S                                   # orient so that high = watermarked
            row = dict(method=method, signal=name, auc=round(a, 4), oriented="-" if a < 0.5 else "+")
            for variant in ("thr", "regime"):
                P = np.zeros_like(G, dtype=bool)
                for k in range(5):
                    tr = fold != k
                    prm = fit(S[tr], G[tr], gsoft[tr], full[tr], variant)
                    P[~tr] = predict(S[~tr], gsoft[~tr], prm)
                m = np.array([metrics.miou_image(P[i], G[i]) for i in range(len(G))])   # official
                row[f"cv_{variant}"] = m.mean()
                row[f"cv_{variant}_full"] = m[full].mean(); row[f"cv_{variant}_partial"] = m[~full].mean()
                if best is None or m.mean() > best["score"]:
                    best = dict(score=m.mean(), signal=name, variant=variant, sign=-1 if a < 0.5 else 1, P=P, m=m)
            rows.append(row); print(row, flush=True)

        # held-out predictions of the best predictor, for the notebook
        np.savez_compressed(data.RESULTS / f"{tag}_cv_{method}.npz", names=np.array(names), pred=best["P"],
                            miou=best["m"], signal=best["signal"], variant=best["variant"])
        # refit on all train, predict test
        S = best["sign"] * sig[best["signal"]]
        prm = fit(S, G, gsoft, full, best["variant"])
        tn, tsig, tgsoft, _ = load(method, "test")
        P = predict(best["sign"] * tsig[best["signal"]], tgsoft, prm)
        test_preds.update({n: p.astype(np.uint8) for n, p in zip(tn, P)})
        print(f"== {method}: best {best['signal']} / {best['variant']} cv={best['score']:.4f} params={prm}", flush=True)
        with open(data.RESULTS / f"{tag}_best_{method}.json", "w") as f:
            json.dump(dict(signal=best["signal"], variant=best["variant"], sign=best["sign"], cv=best["score"],
                           params={k: float(v) for k, v in prm.items()}), f, indent=1)

    # sanity: vectorised mIoU == official
    i = np.random.default_rng(0).integers(0, len(G), 20)
    assert np.allclose(miou_batch(P_ := (S[i] > np.median(S)), G[i]), [metrics.miou_image(P_[j], G[i][j]) for j in range(20)])

    pd.DataFrame(rows).round(4).to_csv(data.RESULTS / f"{tag}_diag.csv", index=False)
    pd.DataFrame(regime_rows).round(4).to_csv(data.RESULTS / f"{tag}_regime.csv", index=False)
    path = metrics.write_submission(test_preds, data.RESULTS / f"sub_{tag}_signals.pt")
    print("wrote", path)


if __name__ == "__main__":
    main()
