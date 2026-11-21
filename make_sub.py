"""Build a submission from per-method probability maps (CV protocol of evalmaps.py, refit on all train).
    python make_sub.py --name step3_probe --pixelseal probe_ens_pixelseal --trustmark probe_ens_trustmark --stegastamp probe_stegastamp
Also writes an averaged map tag when given 'avg:tagA+tagB' (saved as results/preds/<name>_<method>_{train,test}.npz)."""
import argparse

import numpy as np

import data
import metrics
from evalmaps import apply, evaluate, fit_params, gate_scores, load_maps


def averaged(spec, method, name):
    tags = spec[4:].split("+"); tag = f"{name}_{method}"
    for split in ("train", "test"):
        maps = [dict(zip(*load_maps(t, split))) for t in tags]
        names = sorted(maps[0])
        np.savez_compressed(data.RESULTS / "preds" / f"{tag}_{split}.npz", names=np.array(names),
                            prob=np.mean([[m[n] for n in names] for m in maps], 0).astype(np.float16))
    return tag


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
    for m in data.METHODS:
        ap.add_argument(f"--{m}", required=True)
    a = ap.parse_args()
    df = data.train_table().set_index("name"); g = data.train_masks128(); preds, cvs = {}, []
    for m in data.METHODS:
        tag = getattr(a, m)
        if tag.startswith("avg:"):
            tag = averaged(tag, m, a.name)
        row, _, _ = evaluate(tag); cvs.append(row["cv"]); print(m, tag, "CV", round(row["cv"], 4), flush=True)
        names, P = load_maps(tag, "train"); gs = gate_scores(m, "train")
        prm = fit_params(P, np.stack([g[n] for n in names]), np.array([gs[n] for n in names]), df.loc[names].full_watermark.values)
        tn, Pt = load_maps(tag, "test"); gt = gate_scores(m, "test")
        preds.update(zip(tn, apply(Pt, np.array([gt[n] for n in tn]), prm).astype(np.uint8)))
        print("   params", {k: round(float(v), 3) for k, v in prm.items()})
    path = metrics.write_submission(preds, data.RESULTS / f"sub_{a.name}.pt")
    print("CV mean", round(float(np.mean(cvs)), 4), "->", path)


if __name__ == "__main__":
    main()
