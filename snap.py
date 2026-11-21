"""Idea 3: snap a probability map to SAM segments (sam_segments.py).
    python snap.py <tag> [alpha]
For every cell: S = mean of the map over the SMALLEST SAM segment containing it (cells in no segment keep the map);
Q = alpha * P + (1 - alpha) * S.  Writes results/preds/<tag>_snap<alpha*10>_{train,test}.npz (then evaluate with
postproc.py / recteval.py like any map)."""
import sys
import numpy as np
import data
from evalmaps import load_maps


def segments(split):
    z = np.load(data.RESULTS / f"sam_{split}.npz"); off = z["offsets"]; M = z["masks"]
    return {n: np.unpackbits(M[off[i]:off[i + 1]], axis=1)[:, :128 * 128].reshape(-1, 128, 128).astype(bool)
            for i, n in enumerate(z["names"].tolist())}


def snap(p, segs, alpha):
    if len(segs) == 0:
        return p
    area = segs.reshape(len(segs), -1).sum(1); order = np.argsort(-area)          # large first, small overwrite
    S = p.copy()
    for k in order:
        S[segs[k]] = p[segs[k]].mean()
    return alpha * p + (1 - alpha) * S


if __name__ == "__main__":
    tag = sys.argv[1]; alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    for split in ("train", "test"):
        segs = segments(split); names, P = load_maps(tag, split)
        Q = np.stack([snap(p, segs[n], alpha) for n, p in zip(names, P)])
        np.savez_compressed(data.RESULTS / "preds" / f"{tag}_snap{int(alpha * 10)}_{split}.npz", names=np.array(names), prob=Q.astype(np.float16))
    print("done", tag, alpha)
