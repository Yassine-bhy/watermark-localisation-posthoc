"""Step 1b: three more signal maps found while reading the tools and the literature (GPU, via Slurm).
    python signals_b.py --method trustmark --split train
Writes results/signals_b/<method>_<split>.npz (float16, N x 128 x 128):
  det, det512     PixelSeal only: the detector's detection channel (channel 0) per position, at the kit's
                  256 input (8x8) and at native 512 input (16x16); the kit wrapper drops this channel
  agree_hr        PixelSeal / TrustMark: exact agreement map at double input resolution (16x16 instead of 8x8)
  rm              TrustMark only: log energy of the watermark estimated by TrustMark's own remover, x - remove(x)
  mf              all methods: matched filter with the known message. D = embed(x, m) - mean_r embed(x, random_r)
                  is the pattern this message would add; mf = local correlation of high-pass(x) with high-pass(D)
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

import data
from signals import DEV
from wmloc.watermarks import load_watermark

N_RANDOM = 3


def gblur(x, sigma):
    r = int(3 * sigma); k = torch.exp(-torch.arange(-r, r + 1, device=x.device).float() ** 2 / (2 * sigma ** 2))
    k = k / k.sum(); c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def to128(m):
    """1x1xHxW (any H) -> 128x128 numpy."""
    if m.shape[-1] != 128:
        m = F.interpolate(m, size=(128, 128), mode="bilinear", align_corners=False) if m.shape[-1] < 128 else F.avg_pool2d(m, m.shape[-1] // 128)
    return m[0, 0].float().cpu().numpy()


def pixelseal_maps(wm, x, s):
    out = {}
    for key, size in (("", 256), ("512", 512)):
        xi = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True) if size != 512 else x
        det = wm.det
        emb = det.pixel_decoder.output_upscaling(det.convnext(det.preprocess(xi)))
        lin = det.pixel_decoder.linear
        pp = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
        out["det" + key] = to128(pp[:, :1])
        if size == 512:
            out["agree_hr"] = to128((pp[:, 1:] * s[None, :, None, None]).mean(1, keepdim=True))
    return out


def trustmark_maps(wm, x, s):
    n = wm.net
    xi = F.interpolate(x, size=(2 * wm.resolution,) * 2, mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    h = n.maxpool(n.relu(n.bn1(n.conv1(xi))))
    h = n.layer4(n.layer3(n.layer2(n.layer1(h))))
    pp = torch.einsum("bchw,oc->bohw", h, n.fc.weight) + n.fc.bias[None, :, None, None]
    out = {"agree_hr": to128((pp * s[None, :, None, None]).mean(1, keepdim=True))}
    tm = wm.tm
    x256 = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False) * 2 - 1
    res = tm.removal(x256).clamp(-1, 1) - x256
    res = F.interpolate(res, size=(512, 512), mode="bilinear", align_corners=False)
    out["rm"] = to128(gblur(torch.log((res ** 2).mean(1, keepdim=True) + 1e-8), 2))
    return out


def matched_filter(wm, img_u8, x, msg, rng):
    """Local correlation between high-pass(x) and high-pass(message-specific embedding pattern)."""
    def pattern(bits):
        return torch.from_numpy(wm.embed(img_u8, bits).astype(np.float32) / 255).permute(2, 0, 1)[None].to(DEV) - x
    d = pattern(msg) - torch.stack([pattern(rng.integers(0, 2, len(msg))) for _ in range(N_RANDOM)]).mean(0)
    hx, hd = x - gblur(x, 2), d - gblur(d, 2)
    num = gblur((hx * hd).sum(1, keepdim=True), 6)
    den = torch.sqrt(gblur((hx ** 2).sum(1, keepdim=True), 6) * gblur((hd ** 2).sum(1, keepdim=True), 6)) + 1e-8
    return to128(num / den)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=data.METHODS)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    ds = data.load_split(a.split, methods=[a.method]); n = a.limit or len(ds)
    wm = load_watermark(a.method, models_dir=data.TASK_DIR / "models", device=DEV)
    rng = np.random.default_rng(0)
    out, names, t0 = {}, [], time.time()
    for i in range(n):
        it = ds[i]; names.append(it["image_name"])
        x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).to(DEV)
        s = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).to(DEV)
        maps = {"pixelseal": lambda: pixelseal_maps(wm, x, s), "trustmark": lambda: trustmark_maps(wm, x, s),
                "stegastamp": lambda: {}}[a.method]()
        maps["mf"] = matched_filter(wm, it["image"], x, it["message"], rng)
        for k, v in maps.items():
            out.setdefault(k, np.zeros((n, 128, 128), np.float16))[i] = v
        if i % 50 == 0:
            print(f"{i}/{n}  {time.time() - t0:.0f}s", flush=True)
    dst = data.RESULTS / "signals_b"; dst.mkdir(parents=True, exist_ok=True)
    f = dst / f"{a.method}_{a.split}{'_limit' if a.limit else ''}.npz"
    np.savez_compressed(f, names=np.array(names), **out); print("wrote", f, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
