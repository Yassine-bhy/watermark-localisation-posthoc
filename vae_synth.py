"""Step 4c data: FLUX autoencoder round trip of the existing synthetic images (results/synth/<method>_<k>), which makes
fresh watermarks look like the real ones (see vae_test.py: decoding stays perfect, the pixel pattern weakens like in
the real data). Writes results/synth_flux/<method>_<k>/{images.npy, masks.npy, meta.json}."""
import shutil, sys, time
import numpy as np, torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file
import data

d = "external/vae/flux/vae"
vae = AutoencoderKL.from_config(AutoencoderKL.load_config(d)); vae.load_state_dict(load_file(f"{d}/diffusion_pytorch_model.safetensors"))
vae = vae.cuda().eval().half()
src_root, dst_root = data.RESULTS / "synth", data.RESULTS / "synth_flux"; t0 = time.time()
for sh in sorted(src_root.iterdir()):
    dst = dst_root / sh.name
    if (dst / "masks.npy").exists():
        continue
    dst.mkdir(parents=True, exist_ok=True)
    X = np.load(sh / "images.npy", mmap_mode="r"); out = np.lib.format.open_memmap(dst / "images.tmp.npy", "w+", np.uint8, X.shape)
    with torch.no_grad():
        for i in range(0, len(X), 16):
            x = torch.from_numpy(np.ascontiguousarray(X[i:i + 16])).cuda().permute(0, 3, 1, 2).half() / 127.5 - 1
            y = vae.decode(vae.encode(x).latent_dist.mode()).sample
            out[i:i + 16] = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    out.flush(); del out
    (dst / "images.tmp.npy").rename(dst / "images.npy")
    shutil.copy(sh / "meta.json", dst / "meta.json"); shutil.copy(sh / "masks.npy", dst / "masks.npy")
    print(sh.name, len(X), f"{time.time() - t0:.0f}s", flush=True)
print("done")
