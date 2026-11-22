"""Full-image maps from another model, used only where the decode gate says 'full'.
    python route_maps.py --method trustmark --base ens2_trustmark --full research/out/e2eRSpre_trustmark,research/out/e2eRSpre0_trustmark --out ensRS_trustmark
Output map = mean(base, full maps...) on images with gate score > the gate threshold fit on all train (best_gate);
= base elsewhere. Full-map files: path prefix + _{train,test}.npz (names, prob) or results/preds tags."""
import argparse
import numpy as np
import data
from diagnostics import best_gate
from evalmaps import gate_scores, load_maps


def load(spec, split):
    p = data.HERE / f"{spec}_{split}.npz" if "/" in spec else data.RESULTS / "preds" / f"{spec}_{split}.npz"
    z = np.load(p); key = "prob"
    return dict(zip(z["names"].tolist(), z[key].astype(np.float32)))


ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--full", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
df = data.train_table().set_index("name")
gtr = gate_scores(a.method, "train"); ntr = list(gtr); thr = best_gate(np.array([gtr[n] for n in ntr]), df.loc[ntr].full_watermark.values)
for split in ("train", "test"):
    names, B = load_maps(a.base, split); F = [load(s, split) for s in a.full.split(",")]; gs = gate_scores(a.method, split)
    out = []; nf = 0
    for n, b in zip(names, B):
        if gs[n] > thr and all(n in f for f in F):
            out.append(np.mean([b] + [f[n] for f in F], 0)); nf += 1
        else:
            out.append(b)
    np.savez_compressed(data.RESULTS / "preds" / f"{a.out}_{split}.npz", names=np.array(names), prob=np.stack(out).astype(np.float16))
    print(split, f"{nf}/{len(names)} images routed to the full-image average", flush=True)
