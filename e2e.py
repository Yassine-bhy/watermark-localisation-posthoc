"""Step 4d: the decoder's own early layers fine-tuned END-TO-END as a segmenter (not frozen as in probe.py).
    python e2e.py --method pixelseal|trustmark [--synth synth_flux] [--pre_steps 6000] [--ft_epochs 30]
Stage A (optional): train backbone + head on the synthetic images (results/<synth>/<method>_*, FLUX round trip).
Stage B: per CV fold, fine-tune from stage A on the real training folds, predict the held-out fold (same folds as
everywhere); finally fine-tune on all real train -> test maps.
Backbone (pretrained decoder weights): PixelSeal ConvNeXt stem + stages 0-1; TrustMark ResNet stem + layer1-2.
Writes results/preds/<tag>_<method>_{train,test}.npz (128 x 128 probabilities)."""
import argparse, copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import data
from probe import loss_fn
from wmloc.watermarks import load_watermark


class StegaConvs(nn.Module):
    """StegaStamp decoder conv stack (dec_conv_1..7) rebuilt in torch with the ONNX weights (STN skipped).
    TF 'SAME' padding for stride 2 = pad right/bottom by 1. Returns the outputs of conv 2, 4, 5, 6, 7."""
    SPEC = [(3, 32, 2), (32, 32, 1), (32, 64, 2), (64, 64, 1), (64, 64, 2), (64, 128, 2), (128, 128, 2)]

    def __init__(self):
        super().__init__()
        import onnx
        from onnx import numpy_helper
        w = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(data.HERE / "external" / "stega_feats.onnx")).graph.initializer}
        self.convs = nn.ModuleList()
        for k, (ci, co, st) in enumerate(self.SPEC, 1):
            c = nn.Conv2d(ci, co, 3, stride=st, padding=0)
            c.weight.data = torch.from_numpy([v for n, v in w.items() if f"dec_conv_{k}/Conv2D" in n][0].copy())
            c.bias.data = torch.from_numpy([v for n, v in w.items() if f"dec_conv_{k}/BiasAdd" in n][0].copy())
            self.convs.append(c)

    def forward(self, x):
        h, outs = x - 0.5, []
        for k, (c, (_, _, st)) in enumerate(zip(self.convs, self.SPEC), 1):
            h = F.relu(c(F.pad(h, (0, 1, 0, 1) if (st == 2 and k < 7) else (1, 1, 1, 1))))
            if k in (2, 4, 5, 6, 7):
                outs.append(h)
        return outs


class Net(nn.Module):
    def __init__(self, wm):
        super().__init__(); self.kind = wm.name
        if wm.name == "pixelseal":
            cn = wm.det.convnext; self.pre = wm.det.preprocess; self.res = 256
            self.body = nn.ModuleList([cn.downsample_layers[0], cn.stages[0], cn.downsample_layers[1], cn.stages[1]]); c = (96, 192)
            if DEEP:
                self.body.extend([cn.downsample_layers[2], cn.stages[2]]); c = (96, 192, 384)
        elif wm.name == "stegastamp":
            self.ss = StegaConvs(); self.res = 400; c = (32, 64, 64, 128, 128)
        else:
            n = wm.net; self.res = 245
            self.stem = nn.Sequential(n.conv1, n.bn1, n.relu, n.maxpool); self.l1, self.l2 = n.layer1, n.layer2; c = (256, 512)
            if DEEP:
                self.l3 = n.layer3; c = (256, 512, 1024)
        if DEEP:                                               # bigger head with a wider receptive field (objects)
            self.head = nn.Sequential(nn.Conv2d(sum(c), 128, 1), nn.GELU(), nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(128, 128, 3, padding=2, dilation=2), nn.GELU(), nn.Conv2d(128, 64, 3, padding=4, dilation=4), nn.GELU(),
                                      nn.Conv2d(64, 64, 3, padding=8, dilation=8), nn.GELU(), nn.Conv2d(64, 1, 1))
        else:
            self.head = nn.Sequential(nn.Conv2d(sum(c), 64, 1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(64, 32, 3, padding=2, dilation=2), nn.GELU(), nn.Conv2d(32, 1, 1))
        self.norm = nn.BatchNorm2d(sum(c))

    def forward(self, x):                                   # x: B x 3 x 512 x 512 in [0, 1]
        x = F.interpolate(x, size=(self.res,) * 2, mode="bilinear", align_corners=False, antialias=True)
        up = lambda f: F.adaptive_avg_pool2d(f, OUT) if f.shape[-1] > OUT else F.interpolate(f, size=(OUT, OUT), mode="bilinear", align_corners=False)
        if self.kind == "stegastamp":
            return self.head(self.norm(torch.cat([up(f) for f in self.ss(x)], 1)))[:, 0]
        if self.kind == "pixelseal":
            h = self.pre(x); h0 = self.body[1](self.body[0](h)); h1 = self.body[3](self.body[2](h0)); fs = [h0, h1]
            if DEEP:
                fs.append(self.body[5](self.body[4](h1)))
        else:
            h = self.stem(x * 2 - 1); h0 = self.l1(h); h1 = self.l2(h0); fs = [h0, h1]
            if DEEP:
                fs.append(self.l3(h1))
        return self.head(self.norm(torch.cat([up(f) for f in fs], 1)))[:, 0]


FLIP = True
SCALEAUG = 0.0               # prob. of magnifying a random 128-256 px crop to 512 (shrunk-watermark scale)
DEEP = False                 # add a deeper decoder layer (PS stage 2 / TM layer3) for context + bigger head
OUT = 64                     # output grid (64, or 128 = the official mask grid)
LOVASZ = False


def _lovasz_grad(g):
    gts = g.sum(); inter = gts - g.cumsum(0); union = gts + (1 - g).cumsum(0)
    j = 1 - inter / union; j[1:] = j[1:] - j[:-1]
    return j


def lovasz_both(lg, gt):
    """Per-image Lovasz-softmax over both classes (watermarked / clean), averaged: a surrogate of the official
    per-image (IoU_0 + IoU_1) / 2."""
    p = torch.sigmoid(lg).flatten(1); g = (gt > 0.5).float().flatten(1); out = 0
    for pp, gg in ((p, g), (1 - p, 1 - g)):
        for i in range(len(pp)):
            if gg[i].sum() == 0:
                out = out + pp[i].mean(); continue          # class absent: penalise predicting it
            e = (gg[i] - pp[i]).abs(); es, o = torch.sort(e, descending=True)
            out = out + torch.dot(es, _lovasz_grad(gg[i][o]))
    return out / (2 * len(p))
BNFREEZE = False             # keep the decoder's own BatchNorm statistics (only head/norm learn running stats)
AMP = torch.bfloat16          # None = full fp32 (watermark features are tiny; bf16 adds ~4% feature noise)


def batches(X, M, bs, steps, rng):
    for _ in range(steps):
        idx = np.sort(rng.choice(len(X), bs, replace=False))
        x = torch.from_numpy(np.stack([X[i] for i in idx])).cuda().permute(0, 3, 1, 2).float() / 255
        g = torch.from_numpy(np.stack([M[i] for i in idx])).cuda().float()
        g = F.avg_pool2d(g[:, None], 128 // OUT)[:, 0] if OUT < 128 else g
        if FLIP and rng.random() < 0.5:
            x, g = x.flip(-1), g.flip(-1)
        if SCALEAUG and rng.random() < SCALEAUG:
            T = int(rng.integers(128, 257)) // 8 * 8; y0, x0 = rng.integers(0, 512 - T + 1, 2)
            x = F.interpolate(x[:, :, y0:y0 + T, x0:x0 + T], size=(512, 512), mode="bilinear", align_corners=False)
            gs = g.shape[-1]; c0, c1, cs = y0 * gs // 512, x0 * gs // 512, max(1, T * gs // 512)
            g = F.interpolate(g[:, None, c0:c0 + cs, c1:c1 + cs], size=(gs, gs), mode="bilinear", align_corners=False)[:, 0]
        yield x, g


def train(net, X, M, steps, lr, bs, seed):
    rng = np.random.default_rng(seed); net.train()
    if BNFREEZE:
        for n_, mod in net.named_modules():
            if isinstance(mod, nn.BatchNorm2d) and not n_.startswith("norm"):
                mod.eval()
    bb = [p for n, p in net.named_parameters() if not n.startswith(("head", "norm"))]
    hd = [p for n, p in net.named_parameters() if n.startswith(("head", "norm"))]
    opt = torch.optim.AdamW([{"params": bb, "lr": lr * 0.1}, {"params": hd, "lr": lr}], weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr * 0.1, lr], total_steps=steps, pct_start=0.1)
    for x, g in batches(X, M, bs, steps, rng):
        with torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None):
            lg = net(x)
        loss = loss_fn(lg.float(), g) + (lovasz_both(lg.float(), g) if LOVASZ else 0); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net.eval()


@torch.no_grad()
def zoom_map(net, img, z):
    """research/tests/test_zoom.py: tiles of side 512/z, each magnified to the network input (shrunk outpainting
    watermarks come back to the scale the network knows); tile maps averaged on the 128 grid."""
    x = torch.from_numpy(np.ascontiguousarray(img)).cuda().permute(2, 0, 1)[None].float() / 255
    T = int(round(512 / z / 8)) * 8; st = T // 2; pos = sorted(set(list(range(0, 512 - T + 1, st)) + [512 - T]))
    tiles = torch.cat([x[:, :, y:y + T, xx:xx + T] for y in pos for xx in pos]); out = []
    for i in range(0, len(tiles), 16):
        with torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None):
            out.append(torch.sigmoid(net(tiles[i:i + 16]).float()))
    P = F.interpolate(torch.cat(out)[:, None], size=(T // 4, T // 4), mode="area")[:, 0].cpu().numpy()
    acc = np.zeros((128, 128), np.float32); cnt = np.zeros((128, 128), np.float32); k = 0
    for y in pos:
        for xx in pos:
            acc[y // 4:y // 4 + T // 4, xx // 4:xx // 4 + T // 4] += P[k]; cnt[y // 4:y // 4 + T // 4, xx // 4:xx // 4 + T // 4] += 1; k += 1
    return acc / np.maximum(cnt, 1)


@torch.no_grad()
def predict_z(net, X, zooms=(2, 3, 4)):
    return np.stack([np.mean([zoom_map(net, img, z) for z in zooms], 0) for img in X])


@torch.no_grad()
def predict_v(net, X):
    """4-way flip TTA (identity, h, v, hv), each un-flipped before averaging."""
    out = []
    for i in range(0, len(X), 16):
        x = torch.from_numpy(np.stack(X[i:i + 16])).cuda().permute(0, 3, 1, 2).float() / 255; p = 0
        with torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None):
            for d in ((), (-1,), (-2,), (-2, -1)):
                xx = x.flip(d) if d else x; q = torch.sigmoid(net(xx).float()); p = p + (q.flip(d) if d else q)
        out.append(F.interpolate((p / 4)[:, None], size=(128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict(net, X):
    out = []
    for i in range(0, len(X), 16):
        x = torch.from_numpy(np.stack(X[i:i + 16])).cuda().permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None):
            p = torch.sigmoid(net(x).float())
            if FLIP:
                p = (p + torch.sigmoid(net(x.flip(-1)).float()).flip(-1)) / 2
        out.append(F.interpolate(p[:, None], size=(128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--synth", default="synth_flux")
    ap.add_argument("--pre_steps", type=int, default=6000); ap.add_argument("--ft_epochs", type=int, default=30)
    ap.add_argument("--noflip", type=int, default=0, help="no left-right flips (they mirror the watermark pattern)")
    ap.add_argument("--seed", type=int, default=0, help="seed offset (seed ensembles)")
    ap.add_argument("--scaleaug", type=float, default=0.0)
    ap.add_argument("--kinds", default="", help="synthetic kinds to keep, e.g. holes,removal,addition")
    ap.add_argument("--realfull", type=int, default=0, help="fine-tune on real FULL images only")
    ap.add_argument("--zoom", type=int, default=1, help="also write zoomed-tile maps <tag>Z (mean of zooms 2,3,4)")
    ap.add_argument("--deep", type=int, default=0); ap.add_argument("--out", type=int, default=64); ap.add_argument("--foldens", type=int, default=0, help="test map = mean of the 5 fold models + full model")
    ap.add_argument("--bnfreeze", type=int, default=0); ap.add_argument("--lovasz", type=int, default=0)
    ap.add_argument("--fp32", type=int, default=0, help="no bf16 autocast")
    ap.add_argument("--res", type=int, default=0, help="input resolution (default: the decoder's own, 256/245/400)")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--tag", default="e2e")
    a = ap.parse_args(); t0 = time.time(); torch.manual_seed(a.seed)
    global FLIP, AMP, BNFREEZE, LOVASZ, OUT, DEEP, SCALEAUG
    SCALEAUG = a.scaleaug
    OUT = a.out; DEEP = bool(a.deep)
    LOVASZ = bool(a.lovasz)
    BNFREEZE = bool(a.bnfreeze)
    FLIP = not a.noflip
    AMP = None if a.fp32 else torch.bfloat16
    wm = load_watermark(a.method, models_dir=data.TASK_DIR / "models", device="cuda")
    net = Net(wm).cuda().float()
    if a.res:
        net.res = a.res
    tr = data.load_split("train", [a.method]); names = [tr[i]["image_name"] for i in range(len(tr))]
    Xr = [tr[i]["image"] for i in range(len(tr))]; g = data.train_masks128(); Mr = [g[n] for n in names]
    fold = data.train_table().set_index("name").loc[names].fold.values
    isfull = data.train_table().set_index("name").loc[names].full_watermark.values
    if a.pre_steps:
        sh = sorted(p for d in a.synth.split(",") for p in (data.RESULTS / d).glob(f"{a.method}_*") if (p / "images.npy").exists() and (p / "masks.npy").exists())
        Xs, Ms = [], []
        for d in sh:
            X = np.load(d / "images.npy", mmap_mode="r"); Mm = np.load(d / "masks.npy")
            if a.kinds:
                if (d / "prep.npz").exists():
                    kinds = np.load(d / "prep.npz")["kind"]
                else:
                    import json
                    kinds = np.array([r["kind"] for r in json.loads((d / "meta.json").read_text())["samples"]])
                keep = np.isin(kinds, a.kinds.split(","))
            else:
                keep = np.ones(len(X), bool)
            Xs += [X[j] for j in np.nonzero(keep)[0]]; Ms += [Mm[j] for j in np.nonzero(keep)[0]]
        print("synthetic", len(Xs), flush=True)
        net = train(net, Xs, Ms, a.pre_steps, a.lr, a.bs, 100 * a.seed)
        zs = predict(net, Xr); np.savez_compressed(data.RESULTS / "preds" / f"{a.tag}pre_{a.method}_train.npz", names=np.array(names), prob=zs.astype(np.float16))
        print("stage A done (zero-shot maps on real train saved)", f"{time.time() - t0:.0f}s", flush=True)
    base = copy.deepcopy(net.state_dict()); held = np.zeros((len(names), 128, 128), np.float32); held_v = np.zeros_like(held); held_z = np.zeros_like(held)
    te_ds = data.load_split("test", [a.method]); tn = [te_ds[i]["image_name"] for i in range(len(te_ds))]
    Xt = [te_ds[i]["image"] for i in range(len(te_ds))]; test_sum = np.zeros((len(tn), 128, 128), np.float32); n_models = 0
    ft_steps = lambda n: a.ft_epochs * n // a.bs
    for k in range(5):
        net.load_state_dict(base); idx = np.nonzero((fold != k) & (isfull if a.realfull else True))[0]
        train(net, [Xr[i] for i in idx], [Mr[i] for i in idx], ft_steps(len(idx)), a.lr * 0.3, a.bs, 100 * a.seed + k + 1)
        te = np.nonzero(fold == k)[0]; held[te] = predict(net, [Xr[i] for i in te]); held_v[te] = predict_v(net, [Xr[i] for i in te])
        if a.zoom:
            held_z[te] = predict_z(net, [Xr[i] for i in te])
        if a.foldens:
            test_sum += predict(net, Xt); n_models += 1
        print("fold", k, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(data.RESULTS / "preds" / f"{a.tag}_{a.method}_train.npz", names=np.array(names), prob=held.astype(np.float16))
    np.savez_compressed(data.RESULTS / "preds" / f"{a.tag}V_{a.method}_train.npz", names=np.array(names), prob=held_v.astype(np.float16))
    if a.zoom:
        np.savez_compressed(data.RESULTS / "preds" / f"{a.tag}Z_{a.method}_train.npz", names=np.array(names), prob=held_z.astype(np.float16))
    allidx = np.nonzero(isfull)[0] if a.realfull else np.arange(len(Xr))
    net.load_state_dict(base); train(net, [Xr[i] for i in allidx], [Mr[i] for i in allidx], ft_steps(len(allidx)), a.lr * 0.3, a.bs, 100 * a.seed + 9)
    test_sum += predict(net, Xt); n_models += 1
    np.savez_compressed(data.RESULTS / "preds" / f"{a.tag}_{a.method}_test.npz", names=np.array(tn), prob=(test_sum / n_models).astype(np.float16))
    np.savez_compressed(data.RESULTS / "preds" / f"{a.tag}V_{a.method}_test.npz", names=np.array(tn), prob=predict_v(net, Xt).astype(np.float16))
    if a.zoom:
        np.savez_compressed(data.RESULTS / "preds" / f"{a.tag}Z_{a.method}_test.npz", names=np.array(tn), prob=predict_z(net, Xt).astype(np.float16))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
