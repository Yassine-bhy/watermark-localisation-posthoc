"""Step 1: per-image signal maps on the official 128x128 grid (GPU, run through Slurm).
    python signals.py --method pixelseal --split train
Writes results/signals/<method>_<split>.npz with, for every image (float16, N x 128 x 128 unless noted):
  resid        log residual energy: r = x - median3x3(x), mean r^2 per 4x4 cell, Gaussian-smoothed
  agree        known-message agreement a = mean_k (2m_k-1) * l_k(i,j) from the extractor's per-position logits
               (exact for PixelSeal and TrustMark, 8x8 upsampled; absent for StegaStamp: nonlinear head)
  win64/128/256  sliding-window decoding: decode square crops (stride = half size) and score
               mean_k (2m_k-1) * tanh(l_k); each pixel gets the mean over the crops that cover it
  global_soft, global_acc (N,)  whole-image agreement and bit accuracy (used to tell full from partial)
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

import data
from wmloc.watermarks import load_watermark

DEV = "cuda"


def residual_energy(x):
    """x: 1x3x512x512 in [0,1] -> 128x128 log residual energy, smoothed (sigma = 1 cell)."""
    p = F.pad(x, (1, 1, 1, 1), mode="reflect")
    med = p.unfold(2, 3, 1).unfold(3, 3, 1).reshape(*x.shape, 9).median(-1).values
    e = ((x - med) ** 2).mean(1, keepdim=True)                       # 1x1x512x512
    e = F.avg_pool2d(e, 4)                                           # 1x1x128x128 (4x4 cells)
    k = torch.exp(-torch.arange(-3, 4, device=x.device).float() ** 2 / 2)
    k = (k / k.sum())
    e = F.conv2d(F.pad(e, (3, 3, 0, 0), mode="reflect"), k.view(1, 1, 1, 7))
    e = F.conv2d(F.pad(e, (0, 0, 3, 3), mode="reflect"), k.view(1, 1, 7, 1))
    return torch.log(e + 1e-8)[0, 0]


def agreement_map(wm, x, s):
    """Exact per-position agreement (PixelSeal / TrustMark): 8x8 -> 128x128 bilinear."""
    ppl = wm.features(x)["per_position_logits"].float()              # 1 x nbits x h x w
    a = (ppl * s[None, :, None, None]).mean(1, keepdim=True)
    return F.interpolate(a, size=(128, 128), mode="bilinear", align_corners=False)[0, 0]


def window_map(wm, x, s, size, bs=128):
    """Decode every size x size crop (stride size/2); per-pixel mean of the crops' soft agreement."""
    pos = list(range(0, 512 - size + 1, size // 2))
    boxes = [(y, xx) for y in pos for xx in pos]
    crops = torch.cat([x[:, :, y:y + size, xx:xx + size] for y, xx in boxes])
    lg = torch.cat([wm.logits(crops[i:i + bs]).to(DEV) for i in range(0, len(crops), bs)])
    ag = (torch.tanh(lg) * s[None]).mean(1)
    acc = torch.zeros(512, 512, device=DEV); cnt = torch.zeros(512, 512, device=DEV)
    for (y, xx), v in zip(boxes, ag):
        acc[y:y + size, xx:xx + size] += v; cnt[y:y + size, xx:xx + size] += 1
    return F.avg_pool2d((acc / cnt)[None, None], 4)[0, 0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=data.METHODS)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick test)")
    a = ap.parse_args()
    ds = data.load_split(a.split, methods=[a.method])
    n = a.limit or len(ds)
    wm = load_watermark(a.method, models_dir=data.TASK_DIR / "models", device=DEV)
    keys = ["resid", "win64", "win128", "win256"] + (["agree"] if a.method != "stegastamp" else [])
    out = {k: np.zeros((n, 128, 128), np.float16) for k in keys}
    out["global_soft"] = np.zeros(n, np.float32); out["global_acc"] = np.zeros(n, np.float32)
    names = []
    t0 = time.time()
    for i in range(n):
        it = ds[i]; names.append(it["image_name"])
        x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).to(DEV)
        s = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).to(DEV)
        out["resid"][i] = residual_energy(x).cpu().numpy()
        if "agree" in out:
            out["agree"][i] = agreement_map(wm, x, s).cpu().numpy()
        for size in (64, 128, 256):
            out[f"win{size}"][i] = window_map(wm, x, s, size).cpu().numpy()
        lg = wm.logits(x).to(DEV)[0]
        out["global_soft"][i] = (torch.tanh(lg) * s).mean().item()
        out["global_acc"][i] = ((lg > 0).float() == (s > 0).float()).float().mean().item()
        if i % 50 == 0:
            print(f"{i}/{n}  {time.time() - t0:.0f}s", flush=True)
    dst = data.RESULTS / "signals"; dst.mkdir(parents=True, exist_ok=True)
    f = dst / f"{a.method}_{a.split}{'_limit' if a.limit else ''}.npz"
    np.savez_compressed(f, names=np.array(names), **out)
    print("wrote", f, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
