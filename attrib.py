"""Decoder attribution maps: which pixels does the decoder use to read the known message?
  occlusion: blur one patch (removes the fine watermark texture, keeps the content), decode the whole image,
             drop = score(original) - score(occluded); score = sum_k (2m_k - 1) * logit_k (linear, no saturation)
  gradient:  d score / d x  times the image's fine detail (x - blur(x)): each pixel's contribution (torch decoders)
Maps are returned on the 128x128 grid."""
import numpy as np
import torch
import torch.nn.functional as F

from signals_b import gblur


def score_fn(wm, s):
    def f(x):                                                   # x: B x 3 x 512 x 512 in [0,1] -> B scores
        return (wm.logits(x).to(x.device).float() * s[None]).sum(1)
    return f


@torch.no_grad()
def occlusion(wm, x, s, patch=32, stride=16, sigma=3.0, bs=64):
    f = score_fn(wm, s); base = f(x)[0]
    xb = gblur(x, sigma)
    pos = list(range(0, 512 - patch + 1, stride))
    boxes = [(y, xx) for y in pos for xx in pos]
    acc = torch.zeros(512, 512, device=x.device); cnt = torch.zeros(512, 512, device=x.device)
    for i in range(0, len(boxes), bs):
        chunk = boxes[i:i + bs]; xs = x.repeat(len(chunk), 1, 1, 1)
        for j, (y, xx) in enumerate(chunk):
            xs[j, :, y:y + patch, xx:xx + patch] = xb[0, :, y:y + patch, xx:xx + patch]
        d = base - f(xs)
        for j, (y, xx) in enumerate(chunk):
            acc[y:y + patch, xx:xx + patch] += d[j]; cnt[y:y + patch, xx:xx + patch] += 1
    return F.avg_pool2d((acc / cnt)[None, None], 4)[0, 0]


def gradient(wm, x, s, sigma=3.0):
    """Input x gradient on the high-frequency part, summed over colour channels."""
    x = x.clone().requires_grad_(True)
    if wm.name == "pixelseal":
        sc = (wm.det(F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True))[:, 1:].float() * s[None]).sum()
    elif wm.name == "trustmark":
        sc = (wm.net(F.interpolate(x, size=(wm.resolution,) * 2, mode="bilinear", align_corners=False, antialias=True) * 2 - 1).float() * s[None]).sum()
    else:
        raise ValueError("no gradients through the StegaStamp ONNX graph")
    g, = torch.autograd.grad(sc, x)
    with torch.no_grad():
        contrib = (g * (x - gblur(x, sigma))).sum(1, keepdim=True)
        return F.avg_pool2d(gblur(contrib, 4), 4)[0, 0]
