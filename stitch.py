"""Step 4b: the decoders' own per-location votes, made dense by shift-and-stitch (exact 64 x 64 maps).
    python stitch.py --method pixelseal|trustmark
PixelSeal / TrustMark end with  mean over locations -> linear, so the global logit of bit k is exactly the mean of
per-location logits l_k(p) = w_k . f_p + b_k.  Known message s = 2m-1 -> agreement a(p) = mean_k s_k l_k(p).
Stride of the last feature grid is 32 px (at 256 px input): shifting the input by 0,4,..,28 px in y and x (64 passes)
and interleaving the 8x8 maps gives 64x64.  Input: image resized to 256 (TrustMark normally uses 245; fully
convolutional, checked by the sanity line).  Writes results/preds/stitch_<method>_{train,test}.npz:
'agree' (raw 64x64), 'prob' (128x128, sigmoid(a / train std) so evalmaps' thresholds apply), 'glob' (global score)."""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
import data
from wmloc.watermarks import load_watermark

SH = list(range(0, 32, 4))


def last_grid(wm, x):
    """x: B x 3 x H x W in [0,1] (H, W multiples of 32) -> per-location bit logits B x nbits x H/32 x W/32."""
    if wm.name == "pixelseal":
        d = wm.det; emb = d.pixel_decoder.output_upscaling(d.convnext(d.preprocess(x))); lin = d.pixel_decoder.linear
        return (torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None])[:, 1:]
    n = wm.net; h = x * 2 - 1
    h = n.maxpool(n.relu(n.bn1(n.conv1(h))))
    for k in ("layer1", "layer2", "layer3", "layer4"):
        h = getattr(n, k)(h)
    return torch.einsum("bchw,oc->bohw", h, n.fc.weight) + n.fc.bias[None, :, None, None]


@torch.no_grad()
def dense_map(wm, img, s):
    x = F.interpolate(img, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
    xs = torch.cat([F.pad(x, (dx, 32 - dx, dy, 32 - dy), mode="reflect") for dy in SH for dx in SH])
    a = (last_grid(wm, xs) * s[None, :, None, None]).mean(1)            # 64 x 9 x 9
    out = torch.zeros(64, 64, device=img.device)
    for q, (dy, dx) in enumerate([(dy, dx) for dy in SH for dx in SH]):
        for i in range(9):
            cy = 32 * i + 16 - dy
            if not 0 <= cy < 256:
                continue
            for j in range(9):
                cx = 32 * j + 16 - dx
                if 0 <= cx < 256:
                    out[cy // 4, cx // 4] = a[q, i, j]
    g = (last_grid(wm, x) * s[None, :, None, None]).mean(1)[0]          # unshifted 8 x 8
    glob = float((wm.logits(img)[0] * s).mean())
    return out, float(g.mean()), glob


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    wm = load_watermark(a.method, models_dir=data.TASK_DIR / "models", device="cuda")
    res = {}
    for split in ("train", "test"):
        ds = data.load_split(split, [a.method]); n = a.limit or len(ds); names, A, G = [], [], []
        for i in range(n):
            it = ds[i]; names.append(it["image_name"])
            img = torch.from_numpy(it["image"].copy()).permute(2, 0, 1)[None].float().div(255).cuda()
            s = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).cuda()
            m, gm, glob = dense_map(wm, img, s); A.append(m.cpu().numpy()); G.append(glob)
            if i < 3:
                print(f"sanity {split} {i}: mean of 8x8 map {gm:.4f}  vs  global agreement (kit logits) {glob:.4f}", flush=True)
        res[split] = (names, np.stack(A), np.array(G)); print(split, n, f"{time.time() - t0:.0f}s", flush=True)
    sd = res["train"][1].std()
    for split, (names, A, G) in res.items():
        P = torch.sigmoid(F.interpolate(torch.from_numpy(A)[:, None] / sd, size=(128, 128), mode="bilinear", align_corners=False))[:, 0]
        np.savez_compressed(data.RESULTS / "preds" / f"stitch_{a.method}_{split}.npz", names=np.array(names),
                            agree=A.astype(np.float32), prob=P.numpy().astype(np.float16), glob=G)
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
