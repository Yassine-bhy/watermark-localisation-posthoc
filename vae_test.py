"""Does a VAE round trip make freshly embedded watermarks look like the real ones?
Real full train images: decode agreement ~0.95-1.0; matched filter PS 0.007, TM 0.002, SS 0.024."""
import numpy as np, torch
from safetensors.torch import load_file
from diffusers import AutoencoderKL
import data
from signals_b import gblur
from wmloc.watermarks import load_watermark
from synth import Coco


def load_vae(name):
    d = {"flux": "external/vae/flux/vae", "sdxl": "external/vae/sdxl"}[name]
    v = AutoencoderKL.from_config(AutoencoderKL.load_config(d)); v.load_state_dict(load_file(f"{d}/diffusion_pytorch_model.safetensors"))
    return v.cuda().eval()


T = lambda a: torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()


def lcorr(x, d):
    hx, hd = x - gblur(x, 2), d - gblur(d, 2)
    num = gblur((hx * hd).sum(1, keepdim=True), 6); den = torch.sqrt(gblur((hx ** 2).sum(1, keepdim=True), 6) * gblur((hd ** 2).sum(1, keepdim=True), 6)) + 1e-8
    return float((num / den).mean())


rng = np.random.default_rng(0); co = Coco(rng)
for vname in ("flux", "sdxl"):
    vae = load_vae(vname)
    def rt(img):
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
        with torch.no_grad():
            y = vae.decode(vae.encode(x).latent_dist.mode()).sample
        return ((y[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    for m in data.METHODS:
        wm = load_watermark(m, models_dir=data.TASK_DIR / "models", device="cuda"); R = []
        for _ in range(6):
            a, _ = co.canvas(); bits = rng.integers(0, 2, wm.nbits); s = torch.tensor(bits * 2 - 1., device="cuda")
            w = rt(wm.embed(a, bits))
            ag = float((torch.tanh(wm.logits(T(w)).cuda()[0]) * s).mean())
            rnd = np.mean([wm.embed(w, rng.integers(0, 2, wm.nbits)).astype(np.float32) for _ in range(2)], 0)
            mf = lcorr(T(w), T(wm.embed(w, bits)) - torch.from_numpy(rnd / 255).permute(2, 0, 1)[None].cuda())
            R.append((ag, mf, float(np.abs(rt(a).astype(float) - a).mean())))
        r = np.mean(R, 0)
        print(f"{vname:5s} {m:10s} after round trip: decode {r[0]:.2f}  matched filter {r[1]:.3f}  | VAE change on clean image {r[2]:.2f}/255", flush=True)
