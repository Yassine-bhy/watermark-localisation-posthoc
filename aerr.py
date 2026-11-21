"""Idea 1 (AEROBLADE-style): autoencoder reconstruction error per pixel, no training.
    python aerr.py
For each VAE in external/vae (flux, sdxl, sd) and every train/test image: e = |x - D(E(x))| (mean over colour),
plus a high-pass version; pooled to 128 x 128. Generated / regenerated regions reconstruct better (lower error).
Writes results/aerr/<vae>_{train,test}.npz with keys names, l1, hp (128 x 128 each)."""
import time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from diffusers import AutoencoderKL
from safetensors.torch import load_file
import data

OUT = data.RESULTS / "aerr"; OUT.mkdir(exist_ok=True)


def load_vae(d):
    d = Path(d); sub = d / "vae" if (d / "vae").exists() else d
    v = AutoencoderKL.from_config(AutoencoderKL.load_config(sub))
    f = next(sub.glob("*.safetensors")); v.load_state_dict(load_file(f)); return v.cuda().eval()


@torch.no_grad()
def main():
    t0 = time.time()
    for vd in sorted(Path("external/vae").iterdir()):
        vae = load_vae(vd)
        for split in ("train", "test"):
            ds = data.load_split(split); names, L1, HP = [], [], []
            for i in range(len(ds)):
                it = ds[i]; names.append(it["image_name"])
                x = torch.from_numpy(it["image"].copy()).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
                y = vae.decode(vae.encode(x).latent_dist.mode()).sample.clamp(-1, 1)
                d = (x - y)
                hp = d - F.avg_pool2d(F.pad(d, (2, 2, 2, 2), mode="reflect"), 5, 1)
                L1.append(F.avg_pool2d(d.abs().mean(1, keepdim=True), 4)[0, 0].cpu().numpy())
                HP.append(F.avg_pool2d(hp.abs().mean(1, keepdim=True), 4)[0, 0].cpu().numpy())
            np.savez_compressed(OUT / f"{vd.name}_{split}.npz", names=np.array(names), l1=np.stack(L1).astype(np.float16), hp=np.stack(HP).astype(np.float16))
            print(vd.name, split, len(names), f"{time.time() - t0:.0f}s", flush=True)
        del vae; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
