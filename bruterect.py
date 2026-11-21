"""Step 4e: brute-force multi-scale rectangle search for partial images (outpainting = watermarked rectangle R).
    python bruterect.py --method trustmark --split train
Score of a window W: S(W) = mean_k s_k * logit_k(crop W resized to the decoder's input), known message s = 2m-1.
Raw logits (no tanh): they keep growing while W gains watermarked area at the right scale and drop when W takes in
clean area or has the wrong size (the pattern was drawn at 256 px and stretched over R), so S peaks at R.
1) coarse grid: heights and widths 48..512 (x1.25), aspect limit of the method, strides h/2, w/2 (>= 8 px);
2) top 5 boxes (non-overlapping) refined by coordinate ascent: each edge +-16, 8, 4, 2 px while S improves.
Run on images whose whole-image score says "partial" (global_soft < 0.6 - loose, the gate decides later).
Writes results/brute_<method>_<split>.npz: names, box (y0, x0, y1, x1), score S, agree (tanh agreement of the box)."""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
import data
from wmloc.watermarks import load_watermark

SIZES = [48]
while SIZES[-1] < 512:
    SIZES.append(min(512, round(SIZES[-1] * 1.25)))
SMALL = [16, 20, 25, 31, 39]          # --small 1: real rectangles go down to ~10 px


class Scorer:
    def __init__(self, method):
        self.m = method
        self.wm = load_watermark(method, models_dir=data.TASK_DIR / "models", device="cuda")
        if method == "stegastamp":
            import onnx
            from onnx2torch import convert
            self.ss = convert(onnx.load(str(data.HERE / "external" / "stega_feats.onnx"))).cuda().eval()
        self.res = {"pixelseal": 256, "trustmark": self.wm.resolution if method == "trustmark" else 0, "stegastamp": 400}[method]

    @torch.no_grad()
    def logits(self, b):                                   # b: B x 3 x res x res in [0, 1]
        if self.m == "pixelseal":
            return self.wm.det(b)[:, 1:].float()
        if self.m == "trustmark":
            return self.wm.net(b * 2 - 1).float()
        out = self.ss(torch.zeros(len(b), 100, device=b.device), b.permute(0, 2, 3, 1).contiguous())
        return out[3].float() if isinstance(out, (list, tuple)) else out.float()

    bias = None                          # per-bit mean logit over the image's grid windows (--center 1)

    @torch.no_grad()
    def logits_all(self, x, boxes, bs=256):
        mode = "bicubic" if self.m == "stegastamp" else "bilinear"; out = []
        for i in range(0, len(boxes), bs):
            b = torch.cat([F.interpolate(x[:, :, y0:y1, x0:x1], size=(self.res, self.res), mode=mode, align_corners=False, antialias=True)
                           for y0, x0, y1, x1 in boxes[i:i + bs]]).clamp(0, 1)
            out.append(self.logits(b))
        return torch.cat(out)

    @torch.no_grad()
    def score(self, x, boxes, s, bs=256):
        if self.bias is not None:
            lg = self.logits_all(x, boxes, bs) - self.bias[None]
            return (lg * s[None]).mean(1).cpu().numpy(), (torch.tanh(lg) * s[None]).mean(1).cpu().numpy()
        S, A = [], []
        mode = "bicubic" if self.m == "stegastamp" else "bilinear"
        for i in range(0, len(boxes), bs):
            b = torch.cat([F.interpolate(x[:, :, y0:y1, x0:x1], size=(self.res, self.res), mode=mode, align_corners=False, antialias=True)
                           for y0, x0, y1, x1 in boxes[i:i + bs]]).clamp(0, 1)
            lg = self.logits(b); S.append((lg * s[None]).mean(1)); A.append((torch.tanh(lg) * s[None]).mean(1))
        return torch.cat(S).cpu().numpy(), torch.cat(A).cpu().numpy()


def grid(lim, sizes=SIZES):
    out = []
    for h in sizes:
        for w in sizes:
            if max(h, w) / min(h, w) > lim:
                continue
            sy, sx = max(8, h // 2), max(8, w // 2)
            ys = sorted(set(list(range(0, 512 - h + 1, sy)) + [512 - h])); xs = sorted(set(list(range(0, 512 - w + 1, sx)) + [512 - w]))
            out += [(y, x_, y + h, x_ + w) for y in ys for x_ in xs]
    return out


def iou(a, b):
    iy = max(0, min(a[2], b[2]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iy * ix
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def ok(b, lim):
    y0, x0, y1, x1 = b; h, w = y1 - y0, x1 - x0
    return y0 >= 0 and x0 >= 0 and y1 <= 512 and x1 <= 512 and h >= 8 and w >= 8 and max(h, w) / min(h, w) <= lim


def refine(sc, x, s, box, best, lim):
    for step in (16, 8, 4, 2):
        while True:
            moves = []
            for k in range(4):
                for d in (-step, step):
                    b = list(box); b[k] += d
                    if ok(b, lim):
                        moves.append(tuple(b))
            if not moves:
                break
            v, _ = sc.score(x, moves, s)
            if v.max() <= best:
                break
            best, box = float(v.max()), moves[int(v.argmax())]
    return box, best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--center", type=int, default=0); ap.add_argument("--small", type=int, default=0); ap.add_argument("--tag", default="brute")
    a = ap.parse_args(); t0 = time.time()
    sc = Scorer(a.method); lim = 2.2 if a.method == "trustmark" else 16.5; G = grid(lim, (SMALL if a.small else []) + SIZES)
    gs = dict(zip(*[np.load(data.RESULTS / "signals" / f"{a.method}_{a.split}.npz")[k].tolist() for k in ("names", "global_soft")]))
    ds = data.load_split(a.split, [a.method]); out = {"names": [], "box": [], "score": [], "agree": []}
    print(a.method, "grid boxes per image:", len(G), flush=True)
    for i, m in enumerate(ds.meta):
        n = m["image_name"]
        if gs[n] > 0.6:
            continue
        it = ds[i]; x = torch.from_numpy(it["image"].copy()).permute(2, 0, 1)[None].float().div(255).cuda()
        s = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).cuda()
        if a.center:
            sc.bias = None; lgG = sc.logits_all(x, G); sc.bias = lgG.mean(0)
            v = ((lgG - sc.bias[None]) * s[None]).mean(1).cpu().numpy(); del lgG
        else:
            v, _ = sc.score(x, G, s)
        top = []
        for j in np.argsort(-v):
            if all(iou(G[j], G[t]) < 0.5 for t in top):
                top.append(j)
            if len(top) == (20 if a.center else 5):
                break
        best = max((refine(sc, x, s, G[j], float(v[j]), lim) for j in top), key=lambda r: r[1])
        _, ag = sc.score(x, [best[0]], s)
        for k, val in zip(("names", "box", "score", "agree"), (n, best[0], best[1], float(ag[0]))):
            out[k].append(val)
        if len(out["names"]) % 10 == 0:
            print(len(out["names"]), f"{time.time() - t0:.0f}s", flush=True)
        if a.limit and len(out["names"]) >= a.limit:
            break
    np.savez(data.RESULTS / f"{a.tag}_{a.method}_{a.split}.npz", **{k: np.array(v) for k, v in out.items()})
    print("done", len(out["names"]), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
