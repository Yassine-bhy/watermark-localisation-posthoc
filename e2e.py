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


class Net(nn.Module):
    def __init__(self, wm):
        super().__init__(); self.kind = wm.name
        if wm.name == "pixelseal":
            cn = wm.det.convnext; self.pre = wm.det.preprocess; self.res = 256
            self.body = nn.ModuleList([cn.downsample_layers[0], cn.stages[0], cn.downsample_layers[1], cn.stages[1]]); c = (96, 192)
        else:
            n = wm.net; self.res = 245
            self.stem = nn.Sequential(n.conv1, n.bn1, n.relu, n.maxpool); self.l1, self.l2 = n.layer1, n.layer2; c = (256, 512)
        self.head = nn.Sequential(nn.Conv2d(sum(c), 64, 1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(64, 32, 3, padding=2, dilation=2), nn.GELU(), nn.Conv2d(32, 1, 1))
        self.norm = nn.BatchNorm2d(sum(c))

    def forward(self, x):                                   # x: B x 3 x 512 x 512 in [0, 1]
        x = F.interpolate(x, size=(self.res,) * 2, mode="bilinear", align_corners=False, antialias=True)
        if self.kind == "pixelseal":
            h = self.pre(x); h0 = self.body[1](self.body[0](h)); h1 = self.body[3](self.body[2](h0))
        else:
            h = self.stem(x * 2 - 1); h0 = self.l1(h); h1 = self.l2(h0)
        up = lambda f: F.adaptive_avg_pool2d(f, 64) if f.shape[-1] > 64 else F.interpolate(f, size=(64, 64), mode="bilinear", align_corners=False)
        return self.head(self.norm(torch.cat([up(h0), up(h1)], 1)))[:, 0]


def batches(X, M, bs, steps, rng):
    for _ in range(steps):
        idx = np.sort(rng.choice(len(X), bs, replace=False))
        x = torch.from_numpy(np.stack([X[i] for i in idx])).cuda().permute(0, 3, 1, 2).float() / 255
        g = F.avg_pool2d(torch.from_numpy(np.stack([M[i] for i in idx])).cuda().float()[:, None], 2)[:, 0]
        if rng.random() < 0.5:
            x, g = x.flip(-1), g.flip(-1)
        yield x, g


def train(net, X, M, steps, lr, bs, seed):
    rng = np.random.default_rng(seed); net.train()
    bb = [p for n, p in net.named_parameters() if not n.startswith(("head", "norm"))]
    hd = [p for n, p in net.named_parameters() if n.startswith(("head", "norm"))]
    opt = torch.optim.AdamW([{"params": bb, "lr": lr * 0.1}, {"params": hd, "lr": lr}], weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr * 0.1, lr], total_steps=steps, pct_start=0.1)
    for x, g in batches(X, M, bs, steps, rng):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(x)
        loss = loss_fn(lg.float(), g); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net.eval()


@torch.no_grad()
def predict(net, X):
    out = []
    for i in range(0, len(X), 16):
        x = torch.from_numpy(np.stack(X[i:i + 16])).cuda().permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = (torch.sigmoid(net(x).float()) + torch.sigmoid(net(x.flip(-1)).float()).flip(-1)) / 2
        out.append(F.interpolate(p[:, None], size=(128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--synth", default="synth_flux")
    ap.add_argument("--pre_steps", type=int, default=6000); ap.add_argument("--ft_epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--tag", default="e2e")
    a = ap.parse_args(); t0 = time.time(); torch.manual_seed(0)
    wm = load_watermark(a.method, models_dir=data.TASK_DIR / "models", device="cuda")
    net = Net(wm).cuda().float()
    tr = data.load_split("train", [a.method]); names = [tr[i]["image_name"] for i in range(len(tr))]
    Xr = [tr[i]["image"] for i in range(len(tr))]; g = data.train_masks128(); Mr = [g[n] for n in names]
    fold = data.train_table().set_index("name").loc[names].fold.values
    if a.pre_steps:
        sh = sorted((data.RESULTS / a.synth).glob(f"{a.method}_*"))
        Xs = [np.load(s / "images.npy", mmap_mode="r") for s in sh]; Ms = [np.load(s / "masks.npy") for s in sh]
        Xs = [x for X in Xs for x in X]; Ms = [m for M in Ms for m in M]   # lists of views (lazy)
        print("synthetic", len(Xs), flush=True)
        net = train(net, Xs, Ms, a.pre_steps, a.lr, a.bs, 0)
        zs = predict(net, Xr); np.savez_compressed(data.RESULTS / "preds" / f"{a.tag}pre_{a.method}_train.npz", names=np.array(names), prob=zs.astype(np.float16))
        print("stage A done (zero-shot maps on real train saved)", f"{time.time() - t0:.0f}s", flush=True)
    base = copy.deepcopy(net.state_dict()); held = np.zeros((len(names), 128, 128), np.float32)
    ft_steps = lambda n: a.ft_epochs * n // a.bs
    for k in range(5):
        net.load_state_dict(base); idx = np.nonzero(fold != k)[0]
        train(net, [Xr[i] for i in idx], [Mr[i] for i in idx], ft_steps(len(idx)), a.lr * 0.3, a.bs, k + 1)
        te = np.nonzero(fold == k)[0]; held[te] = predict(net, [Xr[i] for i in te]); print("fold", k, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(data.RESULTS / "preds" / f"{a.tag}_{a.method}_train.npz", names=np.array(names), prob=held.astype(np.float16))
    net.load_state_dict(base); train(net, Xr, Mr, ft_steps(len(Xr)), a.lr * 0.3, a.bs, 9)
    te = data.load_split("test", [a.method]); tn = [te[i]["image_name"] for i in range(len(te))]
    np.savez_compressed(data.RESULTS / "preds" / f"{a.tag}_{a.method}_test.npz", names=np.array(tn), prob=predict(net, [te[i]["image"] for i in range(len(te))]).astype(np.float16))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
