"""Production-style run of the REAL-DOMAIN synthetic route for FULL images (follow-up of test_realsynth.py, which
passed its kill test for TrustMark).  One GPU, ~15 min for TrustMark / StegaStamp at N=3000, STEPS=6000.
    python research/tests/realsynth_run.py <method> [N=3000] [STEPS=6000] [variant=post] [tag=RS] [seed=0]

1. Synthetic full images from the dataset's own images (see test_realsynth.py): base = real partial image (train +
   gate-partial test, all methods); whole frame watermarked with the kit embedder; hole = real hole mask of a full
   train image of the OTHER two methods (no leak into this method's CV), restored to the base pixels;
   variant post = FLUX-VAE round trip after compositing, none = no round trip, pre = round trip before compositing.
2. e2e.Net (res 512, default head) trained STEPS steps on them -> zero-shot maps.
3. Adaptation to the real images, two recipes compared in the same 5 folds (real FULL images of the training folds):
     ft  = fine-tune on real only (the current e2e.py recipe: 30 epochs, lr 3e-4);
     mix = co-training: about one third real, two thirds synthetic, 500 steps, lr 3e-4.
   Held-out maps are written for ALL train images of the method (full and partial), like results/preds files.
4. The recipe with the better full-regime CV is re-fit on all real full images -> test maps.
Outputs (3 files): research/out/e2e<tag>_<method>_train.npz (names, prob = held-out maps of the chosen recipe,
pre = zero-shot maps), research/out/e2e<tag>_<method>_test.npz (names, prob), same layout as results/preds/*.npz.
All numbers printed are full-regime 5-fold CV with one global threshold fit on the training folds."""
import copy, os, sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch
import data, e2e
from diagnostics import auc, best_gate, miou_batch
from evalmaps import load_maps
from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark

method = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
variant = sys.argv[4] if len(sys.argv) > 4 else "post"; TAG = sys.argv[5] if len(sys.argv) > 5 else "RS"; SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
OUTDIR = os.environ.get("RS_OUT", "research/out"); DRY = bool(os.environ.get("RS_DRY")); MIXSTEPS = int(os.environ.get("RS_MIXSTEPS", 500))
t0 = time.time(); rng = np.random.default_rng(777 + 1000 * SEED + data.METHODS.index(method)); torch.manual_seed(SEED)
e2e.OUT = 64; e2e.DEEP = bool(os.environ.get("RS_DEEP"))
if not torch.cuda.is_bf16_supported():
    e2e.AMP = None
df = data.train_table().set_index("name"); g128 = data.train_masks128(); TS = np.linspace(0.04, 0.96, 24)

# ---- bases and hole masks
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
print(f"{method} [{variant}] N={N} STEPS={STEPS} seed={SEED} deep={e2e.DEEP}: bases {len(bases)}, hole masks {len(holes)}  {time.time() - t0:.0f}s", flush=True)


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
X = np.zeros((N, 512, 512, 3), np.uint8); B = np.zeros((N, 512, 512, 3), np.uint8); H = np.zeros((N, 512, 512), bool); L = np.zeros((N, 128, 128), np.uint8)
for i in range(N):
    b, known, same = bases[rng.integers(len(bases))]; h = sample_hole(known)
    X[i] = wm.embed(b, rng.integers(0, 2, wm.nbits)); B[i] = b; H[i] = h
    lab = ~h
    if known is not None and same: lab = lab | (h & known)
    L[i] = downsample_mask(lab.astype(np.uint8))
    if i % 500 == 0: print(f"   embed {i}/{N}  {time.time() - t0:.0f}s", flush=True)
if variant in ("post", "pre"):
    from diffusers import AutoencoderKL
    from safetensors.torch import load_file
    d = "external/vae/flux/vae"; vae = AutoencoderKL.from_config(AutoencoderKL.load_config(d)); vae.load_state_dict(load_file(f"{d}/diffusion_pytorch_model.safetensors"))
    vae = vae.cuda().eval().half()

    @torch.no_grad()
    def rt_inplace(A):
        for i in range(0, len(A), 16):
            x = torch.from_numpy(A[i:i + 16]).cuda().permute(0, 3, 1, 2).half() / 127.5 - 1
            y = vae.decode(vae.encode(x).latent_dist.mode()).sample
            A[i:i + 16] = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    if variant == "pre": rt_inplace(X)
    np.copyto(X, B, where=H[..., None])
    if variant == "post": rt_inplace(X)
    del vae; torch.cuda.empty_cache()
else:
    np.copyto(X, B, where=H[..., None])
del B, H
print(f"generated {N}  hole area median {np.median(1 - L.reshape(N, -1).mean(1)):.3f}  {time.time() - t0:.0f}s", flush=True)

# ---- real images of this method
tr = data.load_split("train", [method]); names = [tr[i]["image_name"] for i in range(len(tr))]; Xr = [tr[i]["image"] for i in range(len(tr))]
Mr = [g128[n] for n in names]; fold = df.loc[names].fold.values; isfull = df.loc[names].full_watermark.values
fi = np.nonzero(isfull)[0]; G = np.stack([Mr[i] for i in fi]).astype(bool); ff = fold[fi]; hole_frac = 1 - G.reshape(len(fi), -1).mean(1)


def cv_full(Pf, detail=False):
    out = np.zeros(len(fi))
    for k in range(5):
        trn = ff != k; t = max(TS, key=lambda t: miou_batch(Pf[trn] > t, G[trn]).mean()); out[~trn] = miou_batch(Pf[~trn] > t, G[~trn])
    if detail:
        return float(out.mean()), [float(out[(hole_frac >= lo) & (hole_frac < hi)].mean()) for lo, hi in ((0, .03), (.03, .1), (.1, 1))]
    return float(out.mean())


def existing(tag):
    try:
        nm, P = load_maps(tag, "train"); P = dict(zip(nm, P)); return np.stack([P[n] for n in names])
    except Exception:
        return None


ENS = {"pixelseal": "ens3_pixelseal", "trustmark": "ens2_trustmark", "stegastamp": "ens2_stegastamp"}[method]
Pens = existing(ENS)
if Pens is not None:
    c, bins = cv_full(Pens[fi], True); print(f"reference {ENS}: full CV {c:.4f}  by hole size <3% / 3-10% / >10%: {bins[0]:.3f} / {bins[1]:.3f} / {bins[2]:.3f}")

# ---- pretrain on the synthetic images, zero-shot
net = e2e.Net(wm).cuda().float(); net.res = 512
e2e.train(net, X, L, STEPS, 1e-3, 16, 100 * SEED)
pre = e2e.predict(net, Xr); pf = pre[fi]
print(f"zero-shot: hole AUC {np.median([auc(-pf[i].ravel(), ~G[i].ravel()) for i in range(len(fi))]):.3f} | mIoU@best {max(miou_batch(pf > t, G).mean() for t in TS):.3f} | full CV {cv_full(pf):.4f}"
      + (f" | mean({ENS}, zero-shot) {cv_full((Pens[fi] + pf) / 2):.4f}" if Pens is not None else "") + f"   {time.time() - t0:.0f}s", flush=True)
base = copy.deepcopy(net.state_dict())
sub = rng.choice(N, min(N, 1500), replace=False); Xs = [X[j] for j in sub]; Ls = [L[j] for j in sub]


def adapt(recipe, idx, seed):
    net.load_state_dict(base); xr = [Xr[i] for i in idx]; mr = [Mr[i] for i in idx]
    if recipe == "ft":
        e2e.train(net, xr, mr, 20 if DRY else max(20, 30 * len(idx) // 16), 3e-4, 16, seed)
    else:
        rep = max(1, int(round(0.5 * len(Xs) / len(idx))))
        e2e.train(net, xr * rep + Xs, mr * rep + Ls, 20 if DRY else MIXSTEPS, 3e-4, 16, seed)


held = {}
for recipe in ("ft", "mix"):
    hm = np.zeros((len(names), 128, 128), np.float32)
    for k in range(5):
        adapt(recipe, np.nonzero((fold != k) & isfull)[0], 100 * SEED + k + 1)
        te = np.nonzero(fold == k)[0]; hm[te] = e2e.predict(net, [Xr[i] for i in te])
    held[recipe] = hm; c, bins = cv_full(hm[fi], True)
    msg = f"{recipe:3s}: full CV {c:.4f}  by hole size {bins[0]:.3f} / {bins[1]:.3f} / {bins[2]:.3f}"
    if Pens is not None:
        c2, b2 = cv_full((Pens[fi] + hm[fi]) / 2, True); msg += f" | mean({ENS}, new) {c2:.4f}  by hole size {b2[0]:.3f} / {b2[1]:.3f} / {b2[2]:.3f}"
    print(msg + f"   {time.time() - t0:.0f}s", flush=True)
best = max(held, key=lambda r: cv_full(held[r][fi])); print("chosen recipe:", best, flush=True)
if Pens is not None:
    print(f"mean({ENS}, ft, mix): full CV {cv_full((Pens[fi] + held['ft'][fi] + held['mix'][fi]) / 3):.4f} | mean(ft, mix) {cv_full((held['ft'][fi] + held['mix'][fi]) / 2):.4f}", flush=True)

# ---- final model on all real full images -> test maps
adapt(best, fi, 100 * SEED + 9)
te_ds = data.load_split("test", [method]); tn = [te_ds[i]["image_name"] for i in range(len(te_ds))]
pt = e2e.predict(net, [te_ds[i]["image"] for i in range(len(te_ds))])
try:
    np.savez_compressed(f"{OUTDIR}/e2e{TAG}_{method}_train.npz", names=np.array(names), prob=held[best].astype(np.float16), pre=pre.astype(np.float16),
                        other=held["mix" if best == "ft" else "ft"].astype(np.float16))
    np.savez_compressed(f"{OUTDIR}/e2e{TAG}_{method}_test.npz", names=np.array(tn), prob=pt.astype(np.float16))
    print(f"wrote {OUTDIR}/e2e{TAG}_{method}_{{train,test}}.npz", flush=True)
except OSError as e:
    print("maps not saved:", e, flush=True)
print("done", f"{time.time() - t0:.0f}s")
