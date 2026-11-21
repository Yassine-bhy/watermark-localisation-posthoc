"""Decoder-feature probe: a small conv head on the decoder's OWN early feature maps (trained to see its watermark).
    python probe.py --method pixelseal
Features (image resized to the decoder's input size, as the decoder sees it):
  PixelSeal  ConvNeXt stage 0 (96 x 64 x 64) + stage 1 (192 x 32 x 32, upsampled)
  TrustMark  ResNet layer1 (256 x 62 x 62) + layer2 (512 x 31 x 31, upsampled)        -> all at 64 x 64
Head: 1x1 conv -> 64, then two 3x3 convs, -> 1 logit per cell (64 x 64), upsampled to 128.
5-fold CV on the real train images (same folds as everywhere), then a model on all train -> test maps.
Writes results/preds/probe_<method>_train.npz (held-out maps) and probe_<method>_test.npz."""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data
from wmloc.watermarks import load_watermark


SS_LAYERS = [f"gen_decoder/dec_conv_{k}/Relu:0" for k in (2, 4, 5)]
_ss = None


def _to64(f):
    f = f.float()
    return F.adaptive_avg_pool2d(f, 64) if f.shape[-1] > 64 else F.interpolate(f, size=(64, 64), mode="bilinear", align_corners=False)


@torch.no_grad()
def features(wm, x, deep=False):
    global _ss
    if wm.name == "pixelseal":
        cn = wm.det.convnext
        h = wm.det.preprocess(F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True))
        h0 = cn.stages[0](cn.downsample_layers[0](h)); h1 = cn.stages[1](cn.downsample_layers[1](h0))
        fs = [h0, h1] + ([cn.stages[2](cn.downsample_layers[2](h1))] if deep else [])
    elif wm.name == "trustmark":
        n = wm.net
        h = F.interpolate(x, size=(wm.resolution,) * 2, mode="bilinear", align_corners=False, antialias=True) * 2 - 1
        h = n.maxpool(n.relu(n.bn1(n.conv1(h)))); l1 = n.layer1(h); l2 = n.layer2(l1)
        fs = [l1, l2] + ([n.layer3(l2)] if deep else [])
    else:                                                  # StegaStamp: ONNX copy with conv outputs exposed
        import onnxruntime as ort
        if _ss is None:
            _ss = ort.InferenceSession(str(data.HERE / "external" / "stega_feats.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        xi = F.interpolate(x.float(), size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        outs = _ss.run(SS_LAYERS, {"image": xi.permute(0, 2, 3, 1).cpu().numpy(), "secret": np.zeros((1, 100), np.float32)})
        fs = [torch.from_numpy(o).to(x.device) for o in outs]
    return torch.cat([_to64(f) for f in fs], 1).half()


class Head(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, 64, 1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(64, 32, 3, padding=2, dilation=2), nn.GELU(), nn.Conv2d(32, 1, 1))
        self.norm = nn.BatchNorm2d(c)

    def forward(self, f):
        return self.net(self.norm(f.float()))[:, 0]


def loss_fn(lg, gt):
    bce = F.binary_cross_entropy_with_logits(lg, gt); p = torch.sigmoid(lg); d = 0
    for pp, gg in ((p, gt), (1 - p, 1 - gt)):
        d = d + 1 - (2 * (pp * gg).sum((1, 2)) + 1) / (pp.sum((1, 2)) + gg.sum((1, 2)) + 1)
    return bce + 0.5 * d.mean()


def fit(Ftr, Gtr, epochs=60, bs=16, seed=0):
    torch.manual_seed(seed); head = Head(Ftr.shape[1]).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    n = len(Ftr); steps = epochs * ((n + bs - 1) // bs); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(epochs):
        for idx in torch.randperm(n).split(bs):
            f, g = Ftr[idx], Gtr[idx]
            if torch.rand(1) < 0.5:                       # flips act on features and mask alike
                f, g = f.flip(-1), g.flip(-1)
            loss = loss_fn(head(f), g); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return head.eval()


@torch.no_grad()
def predict(head, Fx):
    return torch.cat([F.interpolate(torch.sigmoid(head(Fx[i:i + 32]))[:, None], size=(128, 128), mode="bilinear",
                                    align_corners=False)[:, 0] for i in range(0, len(Fx), 32)]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--deep", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    wm = load_watermark(a.method, models_dir=data.TASK_DIR / "models", device="cuda")
    out = {}
    for split in ("train", "test"):
        ds = data.load_split(split, [a.method]); names, feats = [], []
        for i in range(len(ds)):
            it = ds[i]; names.append(it["image_name"])
            feats.append(features(wm, torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).cuda(), a.deep))
        out[split] = (names, torch.cat(feats))
        print(split, tuple(out[split][1].shape), f"{time.time() - t0:.0f}s", flush=True)
    names, Ftr = out["train"]
    df = data.train_table().set_index("name"); gts = data.train_masks128()
    G64 = torch.stack([F.avg_pool2d(torch.from_numpy(gts[n]).float()[None, None], 2)[0, 0] for n in names]).cuda()
    fold = df.loc[names].fold.values
    held = np.zeros((len(names), 128, 128), np.float32)
    for k in range(5):
        tr = torch.from_numpy(np.nonzero(fold != k)[0]).cuda(); te = np.nonzero(fold == k)[0]
        head = fit(Ftr[tr], G64[tr], a.epochs)
        held[te] = predict(head, Ftr[torch.from_numpy(te).cuda()])
        print("fold", k, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(data.RESULTS / "preds" / f"probe{'_deep' if a.deep else ''}_{a.method}_train.npz", names=np.array(names), prob=held.astype(np.float16))
    head = fit(Ftr, G64, a.epochs)
    tn, Fte = out["test"]
    np.savez_compressed(data.RESULTS / "preds" / f"probe{'_deep' if a.deep else ''}_{a.method}_test.npz", names=np.array(tn), prob=predict(head, Fte).astype(np.float16))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
