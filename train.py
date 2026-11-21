"""Step 2 training (GPU, via Slurm).
  pretrain:  python train.py pretrain --res 256 --epochs 3 --tag wam256
             synthetic data only; after every epoch, scores all 510 real train images (none seen in training)
  finetune:  python train.py finetune --init results/models/wam256_pre.pt --fold 0..4   (held-out fold k)
             python train.py finetune --init ... --fold -1                              (all 510 -> test)
             real train images (folds != k) mixed 1:1 with synthetic ones
Saves models to results/models/, per-image maps (watermark probability + agreement of the predicted bits with the
known message, float16 128x128) to results/preds/<tag>_<split>.npz, and a metrics json next to them."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import data
from metrics import miou_image
from model import BIT_OFF, NBITS, LocNet, load, save

METHODS = list(data.METHODS)
MODELS = data.RESULTS / "models"; PREDS = data.RESULTS / "preds"


def bits_vec(msg):
    v = np.zeros(256, np.float32); b = np.array([int(c) for c in msg], np.float32); v[:len(b)] = b; return v


class Synth(Dataset):
    def __init__(self):
        self.parts = []
        root = Path(os.environ.get("SYNTH_DIR", data.RESULTS / "synth"))   # override for dry runs
        for d in sorted(root.glob("*_*")):
            meta = json.loads((d / "meta.json").read_text())
            self.parts.append((np.load(d / "images.npy", mmap_mode="r"), np.load(d / "masks.npy"),
                               METHODS.index(meta["method"]), [bits_vec(s["message"]) for s in meta["samples"]]))
        self.index = [(p, i) for p, part in enumerate(self.parts) for i in range(len(part[1]))]
        print("synthetic samples:", len(self.index), flush=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        p, i = self.index[j]; im, mk, mi, bits = self.parts[p]
        return torch.from_numpy(np.array(im[i])), torch.from_numpy(mk[i]), mi, torch.from_numpy(bits[i])


class Real(Dataset):
    """Train or test images held in RAM (uint8)."""
    def __init__(self, split, names=None):
        ds = data.load_split(split)
        keep = [i for i, m in enumerate(ds.meta) if names is None or m["image_name"] in names]
        gts = data.train_masks128() if split == "train" else None
        self.names, self.img, self.mask, self.mi, self.bits = [], [], [], [], []
        for i in keep:
            m = ds.meta[i]; n = m["image_name"]
            from PIL import Image
            self.img.append(np.asarray(Image.open(ds.dir / "images" / f"{n}.png").convert("RGB")))
            self.mask.append(gts[n] if gts else np.zeros((128, 128), np.uint8))
            self.mi.append(METHODS.index(m["method"])); self.bits.append(bits_vec(m["message"])); self.names.append(n)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return torch.from_numpy(self.img[i]), torch.from_numpy(self.mask[i]), self.mi[i], torch.from_numpy(self.bits[i])


def loss_fn(det, bits_all, gt, mi, bits):
    gt = gt.float()
    bce = F.binary_cross_entropy_with_logits(det, gt)
    p = torch.sigmoid(det)
    dice = 0
    for pp, gg in ((p, gt), (1 - p, 1 - gt)):
        dice = dice + 1 - (2 * (pp * gg).sum((1, 2)) + 1) / (pp.sum((1, 2)) + gg.sum((1, 2)) + 1)
    seg = bce + 0.5 * dice.mean()
    bl, n = 0.0, 0
    for k, m in enumerate(METHODS):                                # bit loss on watermarked cells only
        sel = mi == k
        if sel.any():
            lg = bits_all[sel, BIT_OFF[m]:BIT_OFF[m] + NBITS[m]]   # b x nb x 128 x 128
            tg = bits[sel, :NBITS[m], None, None].expand_as(lg)
            w = gt[sel][:, None].expand_as(lg)
            if w.sum() > 0:
                bl = bl + (F.binary_cross_entropy_with_logits(lg.float(), tg, reduction="none") * w).sum() / w.sum(); n += 1
    return seg + 0.2 * (bl / max(n, 1)), seg.detach()


@torch.no_grad()
def predict(net, ds, bs=16):
    net.eval(); probs, agree = [], []
    for x, _, mi, bits in DataLoader(ds, batch_size=bs, num_workers=4):
        x = x.cuda().permute(0, 3, 1, 2).float().div(255); mi = mi.cuda(); bits = bits.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            det, ba = net(x, mi)
        probs.append(torch.sigmoid(det.float()).cpu().half())
        for j in range(len(x)):
            m = METHODS[int(mi[j])]; lg = ba[j, BIT_OFF[m]:BIT_OFF[m] + NBITS[m]].float()
            s = bits[j, :NBITS[m], None, None] * 2 - 1
            agree.append((torch.tanh(lg) * s).mean(0).cpu().half())
    net.train()
    return torch.cat(probs).numpy(), torch.stack(agree).numpy()


def score_real(ds, probs, thr=0.5):
    """Official mIoU at threshold thr, per method; also with the best single threshold per method."""
    out = {}
    for k, m in enumerate(METHODS):
        idx = [i for i in range(len(ds)) if ds.mi[i] == k]
        if not idx:
            continue
        at = np.mean([miou_image(probs[i] > thr, ds.mask[i]) for i in idx])
        best = max((np.mean([miou_image(probs[i] > t, ds.mask[i]) for i in idx]), t) for t in np.linspace(0.05, 0.95, 19))
        out[m] = dict(miou_at_05=round(at, 4), best_miou=round(best[0], 4), best_thr=round(best[1], 2))
    out["all_at_05"] = round(float(np.mean([out[m]["miou_at_05"] for m in METHODS if m in out])), 4)
    return out


def run(net, loader, steps, lr, log, eval_fn=None, eval_every=None):
    enc = [p for n, p in net.named_parameters() if n.startswith("enc.")]
    rest = [p for n, p in net.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": lr}, {"params": rest, "lr": lr * 5}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 5], total_steps=steps, pct_start=0.05)
    it, t0, run_l = 0, time.time(), []
    while it < steps:
        for x, gt, mi, bits in loader:
            x = x.cuda(non_blocking=True).permute(0, 3, 1, 2).float().div(255)
            gt, mi, bits = gt.cuda(), mi.cuda(), bits.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                det, ba = net(x, mi)
            loss, seg = loss_fn(det.float(), ba, gt, mi, bits)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            run_l.append(loss.item()); it += 1
            if it % 100 == 0:
                log(dict(step=it, loss=float(np.mean(run_l[-100:])), sec=round(time.time() - t0)))
            if eval_fn and eval_every and it % eval_every == 0:
                eval_fn(it)
            if it >= steps:
                break


class Mixed(Dataset):
    """Each item: a real image (cycled) and a synthetic image alternately -> 1:1 mix."""
    def __init__(self, real, synth, length):
        self.real, self.synth, self.n = real, synth, length; self.rng = np.random.default_rng(0)

    def __len__(self):
        return self.n

    def __getitem__(self, j):
        if j % 2 == 0:
            return self.real[(j // 2) % len(self.real)]
        return self.synth[int(self.rng.integers(len(self.synth)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pretrain", "finetune"])
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tag", default="wam256"); ap.add_argument("--init", default=None)
    ap.add_argument("--fold", type=int, default=-1); ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    torch.manual_seed(0); MODELS.mkdir(parents=True, exist_ok=True); PREDS.mkdir(parents=True, exist_ok=True)
    logf = open(data.RESULTS / f"log_{a.tag}_{a.stage}{'' if a.stage == 'pretrain' else f'_f{a.fold}'}.jsonl", "w")
    def log(d):
        print(d, flush=True); logf.write(json.dumps(d) + "\n"); logf.flush()
    synth = Synth()
    df = data.train_table()

    if a.stage == "pretrain":
        net = LocNet(res=a.res).cuda().train()
        real = Real("train")
        loader = DataLoader(synth, batch_size=a.bs, shuffle=True, num_workers=8, drop_last=True, pin_memory=True, persistent_workers=True)
        steps = int(a.epochs * len(loader))
        def ev(it):
            probs, agree = predict(net, real)
            log(dict(step=it, real_train=score_real(real, probs)))
            np.savez_compressed(PREDS / f"{a.tag}_pre_train.npz", names=np.array(real.names), prob=probs, agree=agree)
            save(net, MODELS / f"{a.tag}_pre.pt", step=it, res=a.res)
        run(net, loader, steps, a.lr, log, ev, eval_every=len(loader) // 2)
        ev(steps)
        return

    # finetune: real folds != k (or all) mixed with synthetic
    net = load(Path(a.init)).train()
    train_names = set(df[df.fold != a.fold].name) if a.fold >= 0 else set(df.name)
    real_tr = Real("train", train_names)
    loader = DataLoader(Mixed(real_tr, synth, a.steps * a.bs), batch_size=a.bs, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    run(net, loader, a.steps, a.lr * 0.5, log)
    if a.fold >= 0:
        held = Real("train", set(df[df.fold == a.fold].name))
        probs, agree = predict(net, held)
        log(dict(fold=a.fold, heldout=score_real(held, probs)))
        np.savez_compressed(PREDS / f"{a.tag}_ft_f{a.fold}.npz", names=np.array(held.names), prob=probs, agree=agree)
    else:
        save(net, MODELS / f"{a.tag}_ft_all.pt", res=net.res)
        test = Real("test")
        probs, agree = predict(net, test)
        np.savez_compressed(PREDS / f"{a.tag}_ft_all_test.npz", names=np.array(test.names), prob=probs, agree=agree)
        log(dict(done="test predictions"))


if __name__ == "__main__":
    main()
