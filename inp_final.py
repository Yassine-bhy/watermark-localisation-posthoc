"""Realistic synthetic data, step 3/3 (main venv): FLUX autoencoder round trip of the inpainted images, labels.
    python inp_final.py <shard dir>   -> images.npy, masks.npy (same format as results/synth_flux)"""
import sys
import numpy as np, torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file
d = sys.argv[1]; X = np.load(f"{d}/inpainted.npy"); z = np.load(f"{d}/prep.npz")
p = "external/vae/flux/vae"; vae = AutoencoderKL.from_config(AutoencoderKL.load_config(p)); vae.load_state_dict(load_file(f"{p}/diffusion_pytorch_model.safetensors"))
vae = vae.cuda().eval().half(); out = np.zeros_like(X)
with torch.no_grad():
    for i in range(0, len(X), 16):
        x = torch.from_numpy(X[i:i + 16]).cuda().permute(0, 3, 1, 2).half() / 127.5 - 1
        y = vae.decode(vae.encode(x).latent_dist.mode()).sample
        out[i:i + 16] = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
np.save(f"{d}/masks.npy", z["label"]); np.save(f"{d}/images.npy.tmp.npy", out)
import os; os.replace(f"{d}/images.npy.tmp.npy", f"{d}/images.npy"); os.remove(f"{d}/inpainted.npy")
print("final done", d)
