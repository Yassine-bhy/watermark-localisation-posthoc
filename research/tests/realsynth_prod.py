"""Deliverable maps of the real-domain synthetic route (RS) for FULL images: zero-shot and fine-tuned models,
held-out for train and final for test, in the layout of results/preds/*.npz.
    python research/tests/realsynth_prod.py <method> <recipe> <steps> [N=1000] [tag=A] [seed=0]
recipe: none | post | du384 | jpg90 | blur07 | str60 | hires1024 | pre | pre2 | prepost | mix:a+b+c
        (see test_realsynth_sim.py; a mix of the non-pre recipes is the robust choice)

Writes (4 files):
  research/out/e2eRSz<tag>_<method>_{train,test}.npz   prob = ZERO-SHOT maps (model never saw a real label)
  research/out/e2eRSf<tag>_<method>_{train,test}.npz   prob = model fine-tuned on real FULL images
        train: held-out 5-fold maps for ALL train images of the method (same folds as everywhere);
        test : model fine-tuned on all real full train images.
The models are holes-only specialists: use these maps ONLY for images the gate calls full
(research/tests/make_ensrs.py builds a drop-in ensemble tag that does exactly that).
Hole masks for the synthetic images come from the full train images of the OTHER two methods (no leak into the CV)."""
import copy, os, sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch
import data, e2e
from diagnostics import auc, best_gate, miou_batch
from evalmaps import load_maps
from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark
from diffusers import AutoencoderKL
from safetensors.torch import load_file

method, RECIPE, STEPS = sys.argv[1], sys.argv[2], int(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
TAG = sys.argv[5] if len(sys.argv) > 5 else "A"; SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
OUTDIR = os.environ.get("RS_OUT", "research/out"); DRY = bool(os.environ.get("RS_DRY"))
t0 = time.time(); rng = np.random.default_rng(12345 + 1000 * SEED + data.METHODS.index(method)); torch.manual_seed(SEED)
e2e.OUT = 64; e2e.DEEP = bool(os.environ.get("RS_DEEP"))
if not torch.cuda.is_bf16_supported():
    e2e.AMP = None
df = data.train_table().set_index("name"); g128 = data.train_masks128(); TS = np.linspace(0.04, 0.96, 24)

tr_all = data.load_split("train"); bases, holes = [], []
for i in range(len(tr_all)):
    it = tr_all[i]
    if not it["full_watermark"]:
        bases.append((it["image"], it["mask"].astype(bool), it["method"] == method))
    elif it["method"] != method:
        holes.append(it["mask"] == 0)
if not DRY:
    for m in data.METHODS:
        ztr = np.load(data.RESULTS / "signals" / f"{m}_train.npz"); zte = np.load(data.RESULTS / "signals" / f"{m}_test.npz")
        gate = best_gate(ztr["global_soft"], df.loc[ztr["names"].tolist()].full_watermark.values)
        part = set(np.array(zte["names"])[zte["global_soft"] < gate].tolist()); te = data.load_split("test", [m])
        for i in range(len(te)):
            if te.meta[i]["image_name"] in part:
                bases.append((te[i]["image"], None, m == method))
print(f"{method} recipe {RECIPE} steps {STEPS} N {N} seed {SEED} deep {e2e.DEEP}: bases {len(bases)}, hole masks {len(holes)}  {time.time() - t0:.0f}s", flush=True)


def aug(h):
    if rng.random() < 0.5: h = h[:, ::-1]
    if rng.random() < 0.5: h = h[::-1]
    if rng.random() < 0.5: h = h.T
    return h


def sample_hole(known):
    for _ in range(8):
        h = aug(holes[rng.integers(len(holes))])
        if rng.random() < 0.25:
            h2 = h | aug(holes[rng.integers(len(holes))])
            if h2.mean() < 0.4: h = h2
        if h.mean() > 0.5: continue
        if known is None or not (h & known).any(): break
    return np.ascontiguousarray(h)


wm = load_watermark(method, models_dir=data.TASK_DIR / "models", device="cuda")
W = np.zeros((N, 512, 512, 3), np.uint8); B = np.zeros((N, 512, 512, 3), np.uint8); H = np.zeros((N, 512, 512), bool); L = np.zeros((N, 128, 128), np.uint8)
MSG = []
for i in range(N):
    b, known, same = bases[rng.integers(len(bases))]; h = sample_hole(known)
    MSG.append(rng.integers(0, 2, wm.nbits)); W[i] = wm.embed(b, MSG[-1]); B[i] = b; H[i] = h
    lab = ~h
    if known is not None and same: lab = lab | (h & known)
    L[i] = downsample_mask(lab.astype(np.uint8))
VAE = []


@torch.no_grad()
def rt(A):
    if not VAE:
        d = "external/vae/flux/vae"; v = AutoencoderKL.from_config(AutoencoderKL.load_config(d)); v.load_state_dict(load_file(f"{d}/diffusion_pytorch_model.safetensors")); VAE.append(v.cuda().eval().half())
    out = np.zeros_like(A)
    for i in range(0, len(A), 16):
        x = torch.from_numpy(A[i:i + 16]).cuda().permute(0, 3, 1, 2).half() / 127.5 - 1
        y = VAE[0].decode(VAE[0].encode(x).latent_dist.mode()).sample
        out[i:i + 16] = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return out


def apply(v, ids):
    """Recipe v on the images ids.  Every recipe except pre / pre2 treats hole and surroundings alike."""
    import io
    from PIL import Image
    from scipy import ndimage as ndi
    Wi, Bi, Hi = W[ids], B[ids], H[ids]; comp = lambda A: np.where(Hi[..., None], Bi, A)
    if v == "none": return comp(Wi)
    if v == "post": return rt(comp(Wi))                              # FLUX VAE round trip of the composite
    if v == "pre": return comp(rt(Wi))                               # bimodal: check the zero-shot score, discard if < 0.6
    if v == "pre2": return comp(rt(rt(Wi)))
    if v == "prepost": return rt(comp(rt(Wi)))
    if v.startswith("str"):                                          # watermark at xx % strength
        a = float(v[3:]) / 100
        return comp(np.clip(np.rint(Bi.astype(np.float32) + a * (Wi.astype(np.float32) - Bi.astype(np.float32))), 0, 255).astype(np.uint8))
    A = comp(Wi)
    if v.startswith("blur"):                                         # Gaussian blur, sigma = xx / 10 px
        s = float(v[4:]) / 10
        for j in range(len(A)): A[j] = ndi.gaussian_filter(A[j], (s, s, 0))
        return A
    if v.startswith("du"):                                           # resize to xx px and back (bicubic)
        r = int(v[2:])
        for j in range(len(A)): A[j] = np.asarray(Image.fromarray(A[j]).resize((r, r), Image.BICUBIC).resize((512, 512), Image.BICUBIC))
        return A
    if v.startswith("jpg"):                                          # JPEG quality xx
        q = int(v[3:])
        for j in range(len(A)):
            buf = io.BytesIO(); Image.fromarray(A[j]).save(buf, "JPEG", quality=q); A[j] = np.asarray(Image.open(buf).convert("RGB"))
        return A
    if v.startswith("hires"):                                        # embed on the base upscaled to xx px, composite, downscale
        r = int(v[5:]) if len(v) > 5 else 1024
        for j, i in enumerate(ids):
            bu = np.asarray(Image.fromarray(B[i]).resize((r, r), Image.BICUBIC)); wu = wm.embed(bu, MSG[i])
            hu = np.asarray(Image.fromarray(H[i].astype(np.uint8) * 255).resize((r, r), Image.NEAREST)) > 127
            A[j] = np.asarray(Image.fromarray(np.where(hu[..., None], bu, wu)).resize((512, 512), Image.LANCZOS))
        return A
    raise ValueError(v)


parts = RECIPE[4:].split("+") if RECIPE.startswith("mix:") else [RECIPE]
pick = np.random.default_rng(7 + SEED).integers(0, len(parts), N); X = np.empty_like(W)
for j, p in enumerate(parts):
    ids = np.nonzero(pick == j)[0]
    if len(ids): X[ids] = apply(p, ids)
del VAE[:], W, B, H; torch.cuda.empty_cache()
print(f"generated {N}  hole area median {np.median(1 - L.reshape(N, -1).mean(1)):.3f}  {time.time() - t0:.0f}s", flush=True)

tr = data.load_split("train", [method]); names = [tr[i]["image_name"] for i in range(len(tr))]; Xr = [tr[i]["image"] for i in range(len(tr))]
Mr = [g128[n] for n in names]; fold = df.loc[names].fold.values; isfull = df.loc[names].full_watermark.values
fi = np.nonzero(isfull)[0]; G = np.stack([Mr[i] for i in fi]).astype(bool); ff = fold[fi]
te_ds = data.load_split("test", [method]); tn = [te_ds[i]["image_name"] for i in range(len(te_ds))]; Xt = [te_ds[i]["image"] for i in range(len(te_ds))]


def cv_full(Pf):
    out = np.zeros(len(fi))
    for k in range(5):
        trn = ff != k; t = max(TS, key=lambda t: miou_batch(Pf[trn] > t, G[trn]).mean()); out[~trn] = miou_batch(Pf[~trn] > t, G[~trn])
    return float(out.mean())


net = e2e.Net(wm).cuda().float(); net.res = int(os.environ.get("RS_RES", 512))
e2e.train(net, X, L, STEPS, 1e-3, 16, 100 * SEED); del X
zs_tr = e2e.predict(net, Xr); zs_te = e2e.predict(net, Xt); pf = zs_tr[fi]
print(f"zero-shot: hole AUC {np.median([auc(-pf[i].ravel(), ~G[i].ravel()) for i in range(len(fi))]):.3f} | mIoU@best {max(miou_batch(pf > t, G).mean() for t in TS):.3f} | full CV {cv_full(pf):.4f}   {time.time() - t0:.0f}s", flush=True)
base = copy.deepcopy(net.state_dict()); held = np.zeros((len(names), 128, 128), np.float32)
ft_steps = lambda n: 20 if DRY else max(20, 30 * n // 16)
for k in range(5):
    net.load_state_dict(base); idx = np.nonzero((fold != k) & isfull)[0]
    e2e.train(net, [Xr[i] for i in idx], [Mr[i] for i in idx], ft_steps(len(idx)), 3e-4, 16, 100 * SEED + k + 1)
    te = np.nonzero(fold == k)[0]; held[te] = e2e.predict(net, [Xr[i] for i in te])
net.load_state_dict(base); e2e.train(net, [Xr[i] for i in fi], [Mr[i] for i in fi], ft_steps(len(fi)), 3e-4, 16, 100 * SEED + 9)
ft_te = e2e.predict(net, Xt)
msg = f"fine-tuned: full CV {cv_full(held[fi]):.4f} | mean(zero-shot, fine-tuned) {cv_full((held[fi] + pf) / 2):.4f}"
for ens in {"pixelseal": ("ens3_pixelseal",), "trustmark": ("ens2_trustmark", "ens4_trustmark"), "stegastamp": ("ens2_stegastamp", "ens4_stegastamp")}[method]:
    try:
        nm, P = load_maps(ens, "train"); P = dict(zip(nm, P)); E = np.stack([P[names[i]] for i in fi])
        msg += f" | {ens} {cv_full(E):.4f} -> +zero-shot+fine-tuned {cv_full((E + held[fi] + pf) / 3):.4f}"
    except Exception:
        pass
print(msg + f"   {time.time() - t0:.0f}s", flush=True)
try:
    for kind, a_tr, a_te in (("z", zs_tr, zs_te), ("f", held, ft_te)):
        np.savez_compressed(f"{OUTDIR}/e2eRS{kind}{TAG}_{method}_train.npz", names=np.array(names), prob=a_tr.astype(np.float16))
        np.savez_compressed(f"{OUTDIR}/e2eRS{kind}{TAG}_{method}_test.npz", names=np.array(tn), prob=a_te.astype(np.float16))
    print(f"wrote {OUTDIR}/e2eRS[z|f]{TAG}_{method}_[train|test].npz", flush=True)
except OSError as e:
    print("maps not saved:", e, flush=True)
print("done", f"{time.time() - t0:.0f}s")
