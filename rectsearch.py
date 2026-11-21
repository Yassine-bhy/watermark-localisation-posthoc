"""Decoder-verified rectangle refinement: given a rough box guess, find the box whose crop decodes the known
message best (outpainted images = the whole watermarked image, shrunk into a larger canvas).
    refine(wm, x, s, box, ...) -> (best_box, best_agreement)
1) random multi-start: n_samples boxes whose edges are moved by up to `spread` x the box size around the guess;
2) coordinate ascent: move each edge by +-step (step = 12%, 6%, 3%, 1.5% of the box size), keep improvements."""
import numpy as np
import torch

from crops import agreement


def _clip(b, max_aspect, min_side=8):
    y0, x0, y1, x1 = [int(round(v)) for v in b]
    y0, x0 = max(0, y0), max(0, x0); y1, x1 = min(512, y1), min(512, x1)
    if y1 - y0 < min_side or x1 - x0 < min_side:
        return None
    h, w = y1 - y0, x1 - x0
    if max(h, w) / min(h, w) > max_aspect:
        return None
    return (y0, x0, y1, x1)


@torch.no_grad()
def refine(wm, x, s, box, n_samples=256, spread=0.3, max_aspect=16.5, seed=0):
    rng = np.random.default_rng(seed)
    y0, x0, y1, x1 = box; h, w = y1 - y0, x1 - x0
    cands = {_clip(box, max_aspect)}
    for _ in range(n_samples * 3):
        d = rng.uniform(-spread, spread, 4) * np.array([h, w, h, w])
        cands.add(_clip(np.array(box) + d, max_aspect))
        if len(cands) > n_samples:
            break
    cands = [c for c in cands if c is not None]
    a = agreement(wm, x, cands, s).cpu().numpy()
    best, best_a = cands[int(a.argmax())], float(a.max())
    for frac in (0.12, 0.06, 0.03, 0.015):
        for _ in range(3):                                     # a few passes per step size
            improved = False
            by0, bx0, by1, bx1 = best; bh, bw = by1 - by0, bx1 - bx0
            moves = []
            for k, size in ((0, bh), (1, bw), (2, bh), (3, bw)):
                for sgn in (-1, 1):
                    b = list(best); b[k] += sgn * max(1, round(frac * size)); moves.append(_clip(b, max_aspect))
            moves = [m for m in moves if m is not None]
            if not moves:
                break
            am = agreement(wm, x, moves, s).cpu().numpy()
            if am.max() > best_a:
                best, best_a, improved = moves[int(am.argmax())], float(am.max()), True
            if not improved:
                break
    return best, best_a
