"""Step 2 data: synthetic training images following the organisers' recipe, made with the kit's own embedders
on clean COCO images (val2017, 5k images + instance masks; downloaded to external/coco).
    python synth.py --method trustmark --n 1500 --shard 0
Each sample is one of three edit types (shares as in the real train set):
  holes      (50%) whole image watermarked, then COCO-object-shaped regions restored to the clean original
                   ("removal") or covered by a clean object from another image ("addition"); label = 1 - holes
  transplant (25%) an object cut from a watermarked image, pasted into a clean image; label = the object
  outpaint   (25%) a crop (aspect <= 16:1, TrustMark <= 2:1) watermarked at its own size, shrunk and pasted into a
                   clean canvas (half of the time a blurred upscale of itself, like a real outpainting); label = rect
Then, with prob 0.5, a mild whole-image degradation (JPEG / resize / noise).
Writes results/synth/<method>_<shard>/{images.npy (N,512,512,3) uint8, masks.npy (N,128,128) uint8, meta.json}."""
import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from pycocotools.coco import COCO

import data
from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark

COCO_DIR = data.HERE / "external" / "coco"
S = 512


class Coco:
    def __init__(self, rng):
        self.c = COCO(str(COCO_DIR / "annotations" / "instances_val2017.json"))
        self.ids = [i for i in self.c.getImgIds() if self.c.getAnnIds(imgIds=i, iscrowd=False)]
        self.rng = rng

    def image(self, i):
        return np.asarray(Image.open(COCO_DIR / "val2017" / self.c.imgs[i]["file_name"]).convert("RGB"))

    def canvas(self):
        """Random 512x512 crop-resize of a random COCO image (+ its id and crop box for mapping masks)."""
        i = int(self.rng.choice(self.ids)); im = self.image(i); h, w = im.shape[:2]
        side = int(min(h, w) * self.rng.uniform(0.6, 1.0))
        y, x = self.rng.integers(0, h - side + 1), self.rng.integers(0, w - side + 1)
        return np.asarray(Image.fromarray(im[y:y + side, x:x + side]).resize((S, S), Image.BICUBIC)), (i, y, x, side)

    def object_masks(self, crop, min_frac=0.002, max_frac=0.5):
        """Instance masks of the canvas' own image, mapped into the 512 canvas, with a usable size."""
        i, y, x, side = crop
        out = []
        for a in self.c.loadAnns(self.c.getAnnIds(imgIds=i, iscrowd=False)):
            m = self.c.annToMask(a)[y:y + side, x:x + side]
            m = np.asarray(Image.fromarray(m * 255).resize((S, S), Image.NEAREST)) > 127
            if min_frac <= m.mean() <= max_frac:
                out.append(m)
        return out

    def random_object(self, min_frac=0.002, max_frac=0.4):
        """A random object mask from another image, placed at a random position/scale in the canvas."""
        for _ in range(20):
            i = int(self.rng.choice(self.ids)); anns = self.c.loadAnns(self.c.getAnnIds(imgIds=i, iscrowd=False))
            m = self.c.annToMask(anns[self.rng.integers(len(anns))]).astype(bool)
            ys, xs = np.nonzero(m)
            if len(ys) < 50:
                continue
            m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            target = np.exp(self.rng.uniform(np.log(min_frac), np.log(max_frac))) * S * S
            sc = np.sqrt(target / m.sum()); h, w = [max(4, min(S, int(v * sc))) for v in m.shape]
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
            out = np.zeros((S, S), bool); y, x = self.rng.integers(0, S - h + 1), self.rng.integers(0, S - w + 1)
            out[y:y + h, x:x + w] = m
            return out
        return None


def holes(wm, co, rng, bits):
    a, crop = co.canvas(); wa = wm.embed(a, bits)
    own = co.object_masks(crop) if rng.random() < 0.5 else []
    k = rng.integers(1, 4)
    ms = [own[j] for j in rng.permutation(len(own))[:k]] if own else []
    while len(ms) < k:
        m = co.random_object(max_frac=0.3)
        if m is not None:
            ms.append(m)
    hole = np.any(ms, 0)
    out = wa.copy()
    if rng.random() < 0.5:
        out[hole] = a[hole]                                    # removal: restored to the clean original
    else:
        c, _ = co.canvas(); out[hole] = c[hole]                # addition: clean foreign content
    return out, (~hole).astype(np.uint8), "holes"


def transplant(wm, co, rng, bits):
    for _ in range(20):
        a, crop = co.canvas(); objs = co.object_masks(crop, 0.003, 0.4)
        if objs:
            break
    else:
        return holes(wm, co, rng, bits)
    wa = wm.embed(a, bits); m = objs[rng.integers(len(objs))]
    b, _ = co.canvas()
    if rng.random() < 0.5:                                     # move the object (integer shift keeps the pattern)
        ys, xs = np.nonzero(m)
        dy = rng.integers(-ys.min(), S - ys.max()); dx = rng.integers(-xs.min(), S - xs.max())
        m2 = np.roll(np.roll(m, dy, 0), dx, 1); wa = np.roll(np.roll(wa, dy, 0), dx, 1); m = m2
    out = b.copy(); out[m] = wa[m]
    return out, m.astype(np.uint8), "transplant"


def outpaint(wm, co, rng, bits, max_aspect):
    i = int(rng.choice(co.ids)); im = co.image(i); H, W = im.shape[:2]
    asp = np.exp(rng.uniform(0, np.log(max_aspect)))
    if rng.random() < 0.5:
        asp = 1 / asp                                          # tall instead of wide
    # crop of the source with that aspect, watermarked at its own (native) size
    ch = int(min(H, W / asp) * rng.uniform(0.5, 1.0)); cw = max(8, int(ch * asp)); ch = max(8, ch)
    y, x = rng.integers(0, H - ch + 1), rng.integers(0, W - cw + 1)
    src = np.ascontiguousarray(im[y:y + ch, x:x + cw])
    wsrc = wm.embed(src, bits)
    # target rectangle in the canvas
    for _ in range(50):
        frac = np.exp(rng.uniform(np.log(0.003), np.log(0.4)))
        th = int(round(np.sqrt(frac * S * S / asp))); tw = int(round(th * asp))
        if 8 <= th <= S and 8 <= tw <= S:
            break
    th, tw = min(max(th, 8), S), min(max(tw, 8), S)
    small = np.asarray(Image.fromarray(wsrc).resize((tw, th), Image.LANCZOS))
    if rng.random() < 0.5:                                     # pseudo-outpainting: blurred upscale of the source
        bg = np.asarray(Image.fromarray(src).resize((S, S), Image.BICUBIC).filter(ImageFilter.GaussianBlur(rng.uniform(2, 8))))
        bg = np.clip(bg + rng.normal(0, 3, bg.shape), 0, 255).astype(np.uint8)
    else:
        bg, _ = co.canvas()
    ty, tx = rng.integers(0, S - th + 1), rng.integers(0, S - tw + 1)
    out = bg.copy(); out[ty:ty + th, tx:tx + tw] = small
    lab = np.zeros((S, S), np.uint8); lab[ty:ty + th, tx:tx + tw] = 1
    return out, lab, "outpaint"


def degrade(img, rng):
    r = rng.random()
    if r < 0.17:
        buf = io.BytesIO(); Image.fromarray(img).save(buf, "JPEG", quality=int(rng.integers(80, 99)))
        return np.asarray(Image.open(buf).convert("RGB"))
    if r < 0.34:
        s = int(rng.integers(384, 640))
        return np.asarray(Image.fromarray(img).resize((s, s), Image.BICUBIC).resize((S, S), Image.BICUBIC))
    if r < 0.5:
        return np.clip(img + rng.normal(0, rng.uniform(0.5, 2), img.shape), 0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=data.METHODS)
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--shard", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(1000 * data.METHODS.index(a.method) + a.shard)
    wm = load_watermark(a.method, models_dir=data.TASK_DIR / "models", device="cuda")
    co = Coco(rng)
    max_aspect = 2.0 if a.method == "trustmark" else 16.0
    imgs = np.zeros((a.n, S, S, 3), np.uint8); masks = np.zeros((a.n, 128, 128), np.uint8); meta = []
    t0 = time.time()
    for i in range(a.n):
        bits = rng.integers(0, 2, wm.nbits)
        u = rng.random()
        img, lab, kind = (holes(wm, co, rng, bits) if u < 0.5 else transplant(wm, co, rng, bits) if u < 0.75
                          else outpaint(wm, co, rng, bits, max_aspect))
        imgs[i] = degrade(img, rng); masks[i] = downsample_mask(lab)
        meta.append(dict(kind=kind, message="".join(map(str, bits)), wm_frac=float(masks[i].mean())))
        if i % 100 == 0:
            print(f"{i}/{a.n} {time.time() - t0:.0f}s", flush=True)
    d = Path(os.environ.get("SYNTH_DIR", data.RESULTS / "synth")) / f"{a.method}_{a.shard}"; d.mkdir(parents=True, exist_ok=True)
    np.save(d / "images.npy", imgs); np.save(d / "masks.npy", masks)
    (d / "meta.json").write_text(json.dumps(dict(method=a.method, samples=meta)))
    print("wrote", d, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
