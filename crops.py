"""Decode arbitrary rectangular crops of one image in a batch, for any of the three methods.
Each crop is resized to the decoder's square working resolution exactly as the kit wrappers do, so a crop that
matches the watermarked rectangle reproduces the geometry used at embedding time.
    agreement(wm, x, boxes, s) -> soft agreement mean_k s_k * tanh(logit_k) per box (1 = perfect, 0 = random)."""
import numpy as np
import torch
import torch.nn.functional as F


def _resize(c, size, mode):
    return F.interpolate(c, size=(size, size), mode=mode, align_corners=False, antialias=True)


@torch.no_grad()
def crop_logits(wm, x, boxes, bs=64):
    """x: 1x3xHxW in [0,1] on GPU; boxes: list of (y0, x0, y1, x1) in pixels -> (len(boxes), nbits) logits."""
    out = []
    for i in range(0, len(boxes), bs):
        chunk = boxes[i:i + bs]
        if wm.name == "pixelseal":
            b = torch.cat([_resize(x[:, :, y0:y1, x0:x1], 256, "bilinear") for y0, x0, y1, x1 in chunk])
            out.append(wm.det(b)[:, 1:].float())
        elif wm.name == "trustmark":
            b = torch.cat([_resize(x[:, :, y0:y1, x0:x1], wm.resolution, "bilinear") for y0, x0, y1, x1 in chunk])
            out.append(wm.net(b * 2 - 1).float())
        else:
            b = torch.cat([_resize(x[:, :, y0:y1, x0:x1], 400, "bicubic").clamp(0, 1) for y0, x0, y1, x1 in chunk])
            lg = wm._run(b.permute(0, 2, 3, 1).cpu().numpy(), [wm.LOGITS])[0]
            out.append(torch.from_numpy(lg).to(x.device).float())
    return torch.cat(out)


def agreement(wm, x, boxes, s):
    return (torch.tanh(crop_logits(wm, x, boxes)) * s[None]).mean(1)


def bbox(mask128):
    """Bounding box of a 128-grid mask, in pixels (y0, x0, y1, x1)."""
    ys, xs = np.nonzero(mask128)
    return int(ys.min()) * 4, int(xs.min()) * 4, int(ys.max() + 1) * 4, int(xs.max() + 1) * 4
