"""Real-domain synthetic PARTIAL images (extends research/tests/realsynth_run.py, which does full images).
    python rs_partial.py <method> [N=3000] [variant=post] [shard=0]
Bases = real images whose decode gate says partial (train + test, all methods; mostly clean, already through the
organisers' pipeline). Masks = real partial train masks of the OTHER two methods (no leak into this method's CV),
split into rectangles (mask fills its bounding box) and objects.
  outpaint (50 %): a crop of another real image with the aspect of a real rectangle, watermarked at its own size,
                   shrunk into that real rectangle's position in the base;  label = rectangle.
  transplant (50 %): another real image watermarked whole, pasted into the base through a real object mask
                   (random flips / transpose);  label = object mask.
variant post = FLUX VAE round trip after compositing (won for PixelSeal full images), none = no round trip.
Writes results/synth_rsp/<method>_<shard>/{images.npy, masks.npy, meta.json} (format of results/synth_flux)."""
import json, sys, time
import numpy as np, torch
from PIL import Image
import data
from diagnostics import best_gate
from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark

method = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000; variant = sys.argv[3] if len(sys.argv) > 3 else "post"
shard = int(sys.argv[4]) if len(sys.argv) > 4 else 0; t0 = time.time(); rng = np.random.default_rng(4242 + 100 * shard + data.METHODS.index(method))
df = data.train_table().set_index("name"); max_aspect = 2.0 if method == "trustmark" else 16.0

bases, rects, objs = [], [], []
tr = data.load_split("train")
for i in range(len(tr)):
    it = tr[i]
    if it["full_watermark"]:
        continue
    bases.append(it["image"]); m = it["mask"].astype(bool)
    if it["method"] == method or not m.any():
        continue
    ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if m[y0:y1, x0:x1].mean() > 0.95:
        h, w = y1 - y0, x1 - x0
        if max(h, w) / min(h, w) <= max_aspect:
            rects.append((y0, x0, h, w))
    else:
        objs.append(m)
for m in data.METHODS:
    ztr = np.load(data.RESULTS / "signals" / f"{m}_train.npz"); zte = np.load(data.RESULTS / "signals" / f"{m}_test.npz")
    gate = best_gate(ztr["global_soft"], df.loc[ztr["names"].tolist()].full_watermark.values)
    part = set(np.array(zte["names"])[zte["global_soft"] < gate].tolist()); te = data.load_split("test", [m])
    for i in range(len(te)):
        if te.meta[i]["image_name"] in part:
            bases.append(te[i]["image"])
print(f"{method}: bases {len(bases)}, rectangles {len(rects)}, objects {len(objs)}  {time.time() - t0:.0f}s", flush=True)
wm = load_watermark(method, models_dir=data.TASK_DIR / "models", device="cuda")
X = np.zeros((N, 512, 512, 3), np.uint8); L = np.zeros((N, 128, 128), np.uint8); kinds = []
for i in range(N):
    b = bases[rng.integers(len(bases))].copy(); src = bases[rng.integers(len(bases))]; bits = rng.integers(0, 2, wm.nbits)
    if rng.random() < 0.5 and rects:
        y0, x0, h, w = rects[rng.integers(len(rects))]
        if rng.random() < 0.5:
            y0, x0, h, w = x0, y0, w, h                        # transpose the rectangle
        asp = w / h; ch = int(512 * rng.uniform(0.5, 1.0) / max(1.0, asp)); cw = int(ch * asp)
        ch, cw = max(16, min(ch, 512)), max(16, min(cw, 512))
        cy, cx = rng.integers(0, 512 - ch + 1), rng.integers(0, 512 - cw + 1)
        crop = np.ascontiguousarray(src[cy:cy + ch, cx:cx + cw]); wcrop = wm.embed(crop, bits)
        b[y0:y0 + h, x0:x0 + w] = np.asarray(Image.fromarray(wcrop).resize((w, h), Image.LANCZOS))
        lab = np.zeros((512, 512), bool); lab[y0:y0 + h, x0:x0 + w] = True; kinds.append("outpaint")
    else:
        m = objs[rng.integers(len(objs))]
        if rng.random() < 0.5: m = m[:, ::-1]
        if rng.random() < 0.5: m = m[::-1]
        if rng.random() < 0.5: m = m.T
        m = np.ascontiguousarray(m); ws = wm.embed(src, bits); b[m] = ws[m]; lab = m; kinds.append("transplant")
    X[i] = b; L[i] = downsample_mask(lab.astype(np.uint8))
    if i % 500 == 0:
        print(f"   {i}/{N}  {time.time() - t0:.0f}s", flush=True)
if variant == "post":
    from diffusers import AutoencoderKL
    from safetensors.torch import load_file
    d = "external/vae/flux/vae"; vae = AutoencoderKL.from_config(AutoencoderKL.load_config(d)); vae.load_state_dict(load_file(f"{d}/diffusion_pytorch_model.safetensors"))
    vae = vae.cuda().eval().half()
    with torch.no_grad():
        for i in range(0, N, 16):
            x = torch.from_numpy(X[i:i + 16]).cuda().permute(0, 3, 1, 2).half() / 127.5 - 1
            y = vae.decode(vae.encode(x).latent_dist.mode()).sample
            X[i:i + 16] = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
out = data.RESULTS / "synth_rsp" / f"{method}_{shard}"
for t in range(10):                                            # intermittent project inode quota: retry
    try:
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "images.npy", X); np.save(out / "masks.npy", L)
        (out / "meta.json").write_text(json.dumps(dict(method=method, variant=variant, samples=[dict(kind=k) for k in kinds]))); break
    except OSError as e:
        print("save failed, retry", t, e, flush=True); time.sleep(30)
print("done", N, f"{time.time() - t0:.0f}s", flush=True)
