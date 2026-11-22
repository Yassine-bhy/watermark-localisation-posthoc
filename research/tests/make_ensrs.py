"""Build a drop-in ensemble tag that adds the RS full-image maps to an existing ensemble (CPU, seconds).
    python research/tests/make_ensrs.py <method> <old_tag> <rs_tag[,rs_tag...]> <out_tag> [old_weight=0.333] [full|partial]
e.g. python research/tests/make_ensrs.py trustmark ens2_trustmark e2eRSzA_trustmark,e2eRSfA_trustmark ensRSA_trustmark
The last argument chooses which images get the RS maps: "full" (default, RS full models: realsynth_prod.py) or
"partial" (RS partial models: realsynth_partial.py); chain two calls to apply both.

For every image (train and test):  gate says full  -> old_weight * old map + (1 - old_weight) * mean(RS maps)
                                   gate says partial -> old map unchanged      (RS models are holes-only specialists)
Gate = the pipeline's own (evalmaps.gate_scores, threshold = diagnostics.best_gate on the train images; the GATE_MAP
environment variable is honoured).  The threshold is fit on all train images: a one-parameter leak, negligible.
Reads old maps from results/preds, RS maps from research/out (or results/preds); writes
research/out/<out_tag>_{train,test}.npz in the results/preds layout — copy the two files to results/preds to use the
tag with postproc.py / recteval.py / make_sub2.py / pipecv.py unchanged.
Prints the CV of old and new with the pipeline's protocol (gate + per-regime sigma / op / threshold, no boxes)."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, data, postproc
from diagnostics import best_gate
from evalmaps import gate_scores, load_maps

method, old_tag, rs_tags, out_tag = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
W_OLD = float(sys.argv[5]) if len(sys.argv) > 5 else 1 / 3        # weight of the old map; the RS maps share the rest equally
SIDE = sys.argv[6] if len(sys.argv) > 6 else "full"                # "full": change gate-full images (RS full models);
                                                                   # "partial": change gate-partial images (RS partial models)
df = data.train_table().set_index("name"); g = data.train_masks128()


def load_any(tag, split):
    for root in ("research/out", str(data.RESULTS / "preds")):
        try:
            z = np.load(f"{root}/{tag}_{split}.npz"); return dict(zip(z["names"].tolist(), z["prob"].astype(np.float32)))
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"{tag}_{split}.npz")


def load_old(tag, split):                                          # old tag: results/preds, or research/out (to chain two calls)
    try:
        return load_maps(tag, split)
    except FileNotFoundError:
        z = np.load(f"research/out/{tag}_{split}.npz"); return z["names"].tolist(), z["prob"].astype(np.float32)


gs_tr = gate_scores(method, "train"); ntr, Ptr = load_old(old_tag, "train"); full_true = df.loc[ntr].full_watermark.values
gate = best_gate(np.array([gs_tr[n] for n in ntr]), full_true); out = {}
for split in ("train", "test"):
    names, P = load_old(old_tag, split); gs = gate_scores(method, split); R = [load_any(t, split) for t in rs_tags]
    isf = np.array([gs[n] > gate for n in names]); Q = P.copy()
    for i, n in enumerate(names):
        if isf[i] == (SIDE == "full"):
            Q[i] = W_OLD * P[i] + (1 - W_OLD) * sum(r[n] for r in R) / len(R)
    np.savez_compressed(f"research/out/{out_tag}_{split}.npz", names=np.array(names), prob=Q.astype(np.float16)); out[split] = (names, P, Q, isf)
    print(f"{split}: {len(names)} images, gate-full {int(isf.sum())}, maps changed on the gate-{SIDE} ones  -> research/out/{out_tag}_{split}.npz")


def pipeline_cv(P, names):
    names = np.array(names); fold = df.loc[names].fold.values; full = df.loc[names].full_watermark.values
    G = np.stack([g[n] for n in names]); gsv = np.array([gs_tr[n] for n in names]); S = postproc.table(P, G); held = np.zeros(len(P))
    for k in range(5):
        tr = fold != k; te = ~tr; pf = gsv > best_gate(gsv[tr], full[tr])
        for reg, regp in ((full, pf), (~full, ~pf)):
            m = S[..., tr & reg].mean(-1); s, o, t = np.unravel_index(np.argmax(m), m.shape); held[te & regp] = S[s, o, t, te & regp]
    return held, full


names, P, Q, _ = out["train"]; a, full = pipeline_cv(P, names); b, _ = pipeline_cv(Q, names); rng = np.random.default_rng(0)
sel = full if SIDE == "full" else ~full; d = (b - a)[sel]
boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
print(f"{method}: pipeline CV (map + post-processing, no boxes)   all / full / partial")
print(f"   {old_tag:28s} {a.mean():.4f} / {a[full].mean():.4f} / {a[~full].mean():.4f}")
print(f"   {out_tag:28s} {b.mean():.4f} / {b[full].mean():.4f} / {b[~full].mean():.4f}")
print(f"   {SIDE} images: paired gain {d.mean():+.4f}, bootstrap 95 % interval [{np.quantile(boot, .025):+.4f}, {np.quantile(boot, .975):+.4f}], better / worse {np.mean(d > 1e-9):.2f} / {np.mean(d < -1e-9):.2f}")
