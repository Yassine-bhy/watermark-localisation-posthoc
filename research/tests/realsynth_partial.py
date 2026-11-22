"""Real-domain synthetic data (RS) for PARTIAL images: one model per run, zero-shot + fine-tuned maps, kill test.
    python research/tests/realsynth_partial.py <method> <recipe> <steps> [N=3000] [tag=A] [seed=0]
recipe: none | post | mix:none+post     (processing of the composite, as in realsynth_prod.py; never 'pre')

Same idea as the full-image RS (REPORT_FULL.md), for the other half of the data:
  base   = a real partial image of the OTHER two methods (train: true regime, test: decode gate). It is ~96 % clean and
           came out of the organisers' pipeline; its small watermark belongs to another method, so it is correctly
           "clean" for this method's detector.  (Bases of the same method are not used: their real watermark would be
           labelled clean.)
  rectangle (50 %, "outpainting"): a rectangle with real-like geometry (area 0.3 - 35 % log-uniform, aspect within the
           method's limit, any position). The base's OWN crop is upscaled to 512 px, watermarked with the kit embedder,
           and only the RESIDUAL is shrunk back and added: the watermark is at the shrunk scale of real outpaintings
           and the image content is untouched (real rectangle borders show no seam: research/tests/test_seam.py).
  object (50 %, "transplantation"): a real object-shaped partial mask of another method (flips / transposes).
           Half of them: another base watermarked whole and PASTED through the mask (visible foreign object, as in a
           real transplant); half: the base itself watermarked inside the mask (no visible object: watermark-only cue).
  label  = official 128 grid of the watermarked region.
Model: e2e.Net at 512 input, default head, trained <steps> on N such images (no real label) -> zero-shot maps;
then 5-fold fine-tune on the real PARTIAL images of the training folds -> held-out maps; final model -> test maps.

Kill test (thresholds written before the first run), on the real partial train images of <method>:
  P1 zero-shot : median within-image AUC >= the old synthetic's zero-shot (e2eR512pre) + 0.05, for rectangles or objects;
  P2 fine-tuned: partial CV (one global threshold fit on the training folds) >= e2eR512 + 0.02,
                 or mean(current ensemble, new) >= ensemble + 0.015.
  Drop RS-partial for a method if neither holds.
Writes 4 files: research/out/e2eRPz<tag>_<method>_{train,test}.npz (zero-shot) and e2eRPf<tag>_... (fine-tuned), in the
results/preds layout. The models are partial-only specialists: use the maps only where the gate says partial."""
import copy, os, sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch
from PIL import Image
import data, e2e
from diagnostics import auc, best_gate, miou_batch
from evalmaps import load_maps
from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark
from diffusers import AutoencoderKL
from safetensors.torch import load_file

method, RECIPE, STEPS = sys.argv[1], sys.argv[2], int(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
TAG = sys.argv[5] if len(sys.argv) > 5 else "A"; SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
OUTDIR = os.environ.get("RS_OUT", "research/out"); DRY = bool(os.environ.get("RS_DRY"))
REALWM = bool(os.environ.get("RS_REALWM"))     # add kinds that carry the REAL watermark, cut from real full images of this method
t0 = time.time(); rng = np.random.default_rng(54321 + 1000 * SEED + data.METHODS.index(method)); torch.manual_seed(SEED)
e2e.OUT = 64; e2e.DEEP = False
if not torch.cuda.is_bf16_supported():
    e2e.AMP = None
df = data.train_table().set_index("name"); g128 = data.train_masks128(); TS = np.linspace(0.04, 0.96, 24)
MAXASP = 2.0 if method == "trustmark" else 16.0


def is_rect(m):
    ys, xs = np.nonzero(m)
    return len(ys) > 0 and m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean() > 0.95


# ---- bases (other methods' partial images) and object shapes (other methods' real object masks)
tr_all = data.load_split("train"); bases, objs = [], []
for i in range(len(tr_all)):
    it = tr_all[i]
    if it["full_watermark"] or it["method"] == method:
        continue
    bases.append(it["image"]); m = it["mask"].astype(bool)
    if m.any() and not is_rect(m):
        objs.append(m)
if not DRY:
    for m_ in data.METHODS:
        if m_ == method:
            continue
        ztr = np.load(data.RESULTS / "signals" / f"{m_}_train.npz"); zte = np.load(data.RESULTS / "signals" / f"{m_}_test.npz")
        gate = best_gate(ztr["global_soft"], df.loc[ztr["names"].tolist()].full_watermark.values)
        part = set(np.array(zte["names"])[zte["global_soft"] < gate].tolist()); te = data.load_split("test", [m_])
        for i in range(len(te)):
            if te.meta[i]["image_name"] in part:
                bases.append(te[i]["image"])
srcs = []                                      # (real full image of this method, its watermark mask or None if unknown)
if REALWM:
    for i in range(len(tr_all)):
        it = tr_all[i]
        if it["full_watermark"] and it["method"] == method:
            srcs.append((it["image"], it["mask"].astype(bool)))
    if not DRY:
        ztr = np.load(data.RESULTS / "signals" / f"{method}_train.npz"); zte = np.load(data.RESULTS / "signals" / f"{method}_test.npz")
        gate = best_gate(ztr["global_soft"], df.loc[ztr["names"].tolist()].full_watermark.values)
        fullset = set(np.array(zte["names"])[zte["global_soft"] > gate].tolist()); te = data.load_split("test", [method])
        for i in range(len(te)):
            if te.meta[i]["image_name"] in fullset:
                srcs.append((te[i]["image"], None))        # holes unknown: ~6 % of the pasted area is mislabelled
print(f"{method} recipe {RECIPE} steps {STEPS} N {N} seed {SEED} realwm {REALWM}: bases {len(bases)}, object masks {len(objs)}, real-watermark sources {len(srcs)}  {time.time() - t0:.0f}s", flush=True)


def sample_rect():
    for _ in range(50):
        frac = np.exp(rng.uniform(np.log(0.003), np.log(0.35))); asp = np.exp(rng.uniform(-np.log(MAXASP), np.log(MAXASP)))
        h = int(round(np.sqrt(frac * 512 * 512 / asp))); w = int(round(h * asp))
        if 8 <= h <= 512 and 8 <= w <= 512 and max(h, w) / min(h, w) <= MAXASP * 1.05:
            return int(rng.integers(0, 512 - h + 1)), int(rng.integers(0, 512 - w + 1)), h, w
    return 200, 200, 100, 100


def aug(m):
    if rng.random() < 0.5: m = m[:, ::-1]
    if rng.random() < 0.5: m = m[::-1]
    if rng.random() < 0.5: m = m.T
    return np.ascontiguousarray(m)


wm = load_watermark(method, models_dir=data.TASK_DIR / "models", device="cuda")
X = np.zeros((N, 512, 512, 3), np.uint8); L = np.zeros((N, 128, 128), np.uint8); kinds = []
for i in range(N):
    b = bases[rng.integers(len(bases))].copy(); bits = rng.integers(0, 2, wm.nbits); lab = np.zeros((512, 512), bool)
    if REALWM and rng.random() < 0.5:                                # REAL watermark, from a real full image of this method
        simg, smask = srcs[rng.integers(len(srcs))]; smask = np.ones((512, 512), bool) if smask is None else smask
        if rng.random() < 0.5:                                       # the whole real image shrunk into a rectangle (outpainting)
            y0, x0, h, w = sample_rect()
            b[y0:y0 + h, x0:x0 + w] = np.asarray(Image.fromarray(simg).resize((w, h), Image.LANCZOS))
            lab[y0:y0 + h, x0:x0 + w] = np.asarray(Image.fromarray(smask.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)) > 127
            kinds.append("rect-real")
        else:                                                        # a real watermarked region pasted through an object mask
            m = aug(objs[rng.integers(len(objs))]); b[m] = simg[m]; lab = m & smask; kinds.append("object-real")
    elif rng.random() < 0.5:                                         # rectangle: shrunk watermark, residual only
        y0, x0, h, w = sample_rect(); s = 512 / max(h, w); W_, H_ = max(16, int(round(w * s))), max(16, int(round(h * s)))
        crop = np.ascontiguousarray(b[y0:y0 + h, x0:x0 + w]); up = np.asarray(Image.fromarray(crop).resize((W_, H_), Image.BICUBIC))
        res = wm.embed(up, bits).astype(np.float32) - up.astype(np.float32)
        small = np.stack([np.asarray(Image.fromarray(res[..., c], mode="F").resize((w, h), Image.LANCZOS)) for c in range(3)], -1)
        b[y0:y0 + h, x0:x0 + w] = np.clip(np.rint(crop.astype(np.float32) + small), 0, 255).astype(np.uint8)
        lab[y0:y0 + h, x0:x0 + w] = True; kinds.append("rect")
    else:
        m = aug(objs[rng.integers(len(objs))])
        if rng.random() < 0.5:                                       # transplant: foreign watermarked content pasted
            src = bases[rng.integers(len(bases))]; b[m] = wm.embed(src, bits)[m]; kinds.append("object-paste")
        else:                                                        # watermark only, same content
            b[m] = wm.embed(b, bits)[m]; kinds.append("object-own")
        lab = m
    X[i] = b; L[i] = downsample_mask(lab.astype(np.uint8))
    if i % 500 == 0: print(f"   generate {i}/{N}  {time.time() - t0:.0f}s", flush=True)
parts = RECIPE[4:].split("+") if RECIPE.startswith("mix:") else [RECIPE]; pick = np.random.default_rng(7 + SEED).integers(0, len(parts), N)
if "post" in parts:
    d = "external/vae/flux/vae"; vae = AutoencoderKL.from_config(AutoencoderKL.load_config(d)); vae.load_state_dict(load_file(f"{d}/diffusion_pytorch_model.safetensors")); vae = vae.cuda().eval().half()
    ids = np.nonzero(pick == parts.index("post"))[0]
    with torch.no_grad():
        for j in range(0, len(ids), 16):
            k = ids[j:j + 16]; x = torch.from_numpy(X[k]).cuda().permute(0, 3, 1, 2).half() / 127.5 - 1
            y = vae.decode(vae.encode(x).latent_dist.mode()).sample
            X[k] = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    del vae; torch.cuda.empty_cache()
frac = L.reshape(N, -1).mean(1); kinds = np.array(kinds)
print(f"generated {N}: kinds {dict(zip(*np.unique(kinds, return_counts=True)))}; watermarked area median {np.median(frac):.3f} (q10 {np.quantile(frac, .1):.3f}, q90 {np.quantile(frac, .9):.3f})  {time.time() - t0:.0f}s", flush=True)

# ---- real images of this method
tr = data.load_split("train", [method]); names = [tr[i]["image_name"] for i in range(len(tr))]; Xr = [tr[i]["image"] for i in range(len(tr))]
Mr = [g128[n] for n in names]; fold = df.loc[names].fold.values; isfull = df.loc[names].full_watermark.values
pi = np.nonzero(~isfull)[0]; G = np.stack([Mr[i] for i in pi]).astype(bool); pf_ = fold[pi]
rect_real = np.array([is_rect(tr[i]["mask"].astype(bool)) for i in pi])
te_ds = data.load_split("test", [method]); tn = [te_ds[i]["image_name"] for i in range(len(te_ds))]; Xt = [te_ds[i]["image"] for i in range(len(te_ds))]


def cv_part(P):
    out = np.zeros(len(pi))
    for k in range(5):
        trn = pf_ != k; t = max(TS, key=lambda t: miou_batch(P[trn] > t, G[trn]).mean()); out[~trn] = miou_batch(P[~trn] > t, G[~trn])
    return out


def describe(P):
    a = np.array([auc(P[i].ravel(), G[i].ravel()) for i in range(len(pi))]); c = cv_part(P)
    return (f"AUC rect {np.median(a[rect_real]):.3f} / object {np.median(a[~rect_real]):.3f} | CV partial {c.mean():.4f} "
            f"(rect {c[rect_real].mean():.3f}, object {c[~rect_real].mean():.3f})")


def existing(tag):
    try:
        nm, P = load_maps(tag, "train"); P = dict(zip(nm, P)); return np.stack([P[names[i]] for i in pi])
    except Exception:
        return None


ENS = {"pixelseal": "ens3_pixelseal", "trustmark": "ens2_trustmark", "stegastamp": "ens2_stegastamp"}[method]
print(f"real partial images of {method}: {len(pi)} ({int(rect_real.sum())} rectangles, {int((~rect_real).sum())} objects). Reference maps:")
for tag in (f"e2eR512pre_{method}", f"e2eR512_{method}", ENS, f"e2eR512s1Z_{method}", f"e2eR512Z_{method}"):
    P = existing(tag)
    if P is not None: print(f"   {tag:26s} {describe(P)}", flush=True)
Pens = existing(ENS)

# ---- train on the synthetic images, zero-shot maps (plain and zoomed)
net = e2e.Net(wm).cuda().float(); net.res = int(os.environ.get("RS_RES", 512))
e2e.train(net, X, L, STEPS, 1e-3, 16, 100 * SEED); del X
zs_tr = e2e.predict(net, Xr); zs_te = e2e.predict(net, Xt)
print(f"zero-shot (plain)        {describe(zs_tr[pi])}   {time.time() - t0:.0f}s", flush=True)
zz = e2e.predict_z(net, [Xr[i] for i in pi])
print(f"zero-shot (zoom 2-4 mean) {describe(zz)} | max(plain, zoom) {describe(np.maximum(zs_tr[pi], zz))}   {time.time() - t0:.0f}s", flush=True)

# ---- fine-tune on the real partial images (5 folds), final model, test maps
base = copy.deepcopy(net.state_dict()); held = np.zeros((len(names), 128, 128), np.float32)
ft_steps = lambda n: 20 if DRY else max(20, 30 * n // 16)
for k in range(5):
    net.load_state_dict(base); idx = np.nonzero((fold != k) & ~isfull)[0]
    e2e.train(net, [Xr[i] for i in idx], [Mr[i] for i in idx], ft_steps(len(idx)), 3e-4, 16, 100 * SEED + k + 1)
    te = np.nonzero(fold == k)[0]; held[te] = e2e.predict(net, [Xr[i] for i in te])
net.load_state_dict(base); e2e.train(net, [Xr[i] for i in pi], [Mr[i] for i in pi], ft_steps(len(pi)), 3e-4, 16, 100 * SEED + 9)
ft_te = e2e.predict(net, Xt)
print(f"fine-tuned               {describe(held[pi])}", flush=True)
print(f"mean(zero-shot, fine-tuned) {describe((held[pi] + zs_tr[pi]) / 2)}", flush=True)
if Pens is not None:
    print(f"mean({ENS}, fine-tuned) {describe((Pens + held[pi]) / 2)}", flush=True)
    print(f"mean({ENS}, fine-tuned, zero-shot) {describe((Pens + held[pi] + zs_tr[pi]) / 3)}   {time.time() - t0:.0f}s", flush=True)
try:
    for kind, a_tr, a_te in (("z", zs_tr, zs_te), ("f", held, ft_te)):
        np.savez_compressed(f"{OUTDIR}/e2eRP{kind}{TAG}_{method}_train.npz", names=np.array(names), prob=a_tr.astype(np.float16))
        np.savez_compressed(f"{OUTDIR}/e2eRP{kind}{TAG}_{method}_test.npz", names=np.array(tn), prob=a_te.astype(np.float16))
    print(f"wrote {OUTDIR}/e2eRP[z|f]{TAG}_{method}_[train|test].npz", flush=True)
except OSError as e:
    print("maps not saved:", e, flush=True)
print("done", f"{time.time() - t0:.0f}s")
