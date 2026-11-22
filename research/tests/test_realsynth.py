"""Kill test (1 GPU, <= 14 min per method): REAL-DOMAIN synthetic FULL images.
    python research/tests/test_realsynth.py <method> [N=1200] [STEPS=3000] [variants=post,pre,none]

Why: the zero-shot gap of the current COCO-based synthetic data is large on full images (hole AUC 0.77-0.87 vs
0.87-0.93 after fine-tuning; research/tests/full_checks.py), and the fine-tune only has ~136 real full images per fold.
The real images are generated-looking scenes; the real holes are segment / blob shapes that ignore the visible
content, 59 % of them touch the image border.  The COCO generator matches neither.

Recipe (the docstring of wmloc/data.py, literally: "the whole image was watermarked and a part of it restored to the
clean original"), built from the dataset's own images:
  base b   = a real PARTIAL image (train: true regime; test: decode gate), any method -> ~96 % clean, real pipeline;
  w        = Emb(b, random message) over the whole frame, with the kit embedder of <method>;
  hole H   = a real hole mask (512 px) of a full train image of the OTHER two methods (no leak into the evaluation),
             random flip / transpose, sometimes the union of two; never on the base's known watermarked region;
  variants : post = FLUX-VAE(composite(w, b, H))      composite first, round trip last (what vae_synth.py does)
             pre  = composite(FLUX-VAE(w), b, H)      round trip the watermarked image, restore pristine b after
             none = composite(w, b, H)                no round trip
  label    = official 128 grid of (not H).
Model: e2e.Net at res 512, trained STEPS steps on N such images (holes only), then
  (a) zero-shot on the real FULL train images of <method>;
  (b) 5-fold fine-tune on real full images only (as `e2e.py --realfull 1`), held-out maps.
Same numbers are printed for the existing maps (e2eR512pre / e2efluxpre / e2eFHpre, e2eR512 / e2eFH / ensembles).

Pass thresholds (written before the first run):
  P1 zero-shot : mIoU at the best global threshold on real full images >= best existing zero-shot + 0.03;
  P2 fine-tuned: 5-fold full CV (threshold fit on the training folds) >= e2eR512 + 0.02,
                 or mean(ensemble, new) >= ensemble + 0.015.
  Drop the idea for a method if neither P1 nor P2 holds for any variant.
Writes research/out/realsynth_<variant>_<method>_train.npz (held-out maps; zero-shot maps under key 'pre')."""
import copy, sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch
import data, e2e
from diagnostics import auc, best_gate, miou_batch
from evalmaps import load_maps
from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark

method = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 1200; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
VARIANTS = (sys.argv[4] if len(sys.argv) > 4 else "post,pre,none").split(","); DRY = len(sys.argv) > 5
import os; OUTDIR = os.environ.get("RS_OUT", "research/out")      # RS_OUT=/tmp/... when scratch cannot take new files
t0 = time.time(); rng = np.random.default_rng(12345 + data.METHODS.index(method)); torch.manual_seed(0)
e2e.OUT = 64; e2e.DEEP = False
if DRY or not torch.cuda.is_bf16_supported():
    e2e.AMP = None
df = data.train_table().set_index("name"); g128 = data.train_masks128(); TS = np.linspace(0.04, 0.96, 24)

# ---- bases (real partial images) and hole masks (real full images of the other methods)
tr_all = data.load_split("train"); bases, holes = [], []
for i in range(len(tr_all)):
    it = tr_all[i]
    if not it["full_watermark"]:
        bases.append((it["image"], it["mask"].astype(bool), it["method"] == method))
    elif it["method"] != method:
        holes.append(it["mask"] == 0)
n_train_bases = len(bases)
if not DRY:
    for m in data.METHODS:
        ztr = np.load(data.RESULTS / "signals" / f"{m}_train.npz"); zte = np.load(data.RESULTS / "signals" / f"{m}_test.npz")
        gate = best_gate(ztr["global_soft"], df.loc[ztr["names"].tolist()].full_watermark.values)
        part = set(np.array(zte["names"])[zte["global_soft"] < gate].tolist()); te = data.load_split("test", [m])
        for i in range(len(te)):
            if te.meta[i]["image_name"] in part:
                bases.append((te[i]["image"], None, m == method))
print(f"{method}: bases {len(bases)} ({n_train_bases} train partial + {len(bases) - n_train_bases} test gate-partial), hole masks {len(holes)}  {time.time() - t0:.0f}s", flush=True)


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


# ---- generate (in memory only: the project's inode quota is almost full)
wm = load_watermark(method, models_dir=data.TASK_DIR / "models", device="cuda")
W = np.zeros((N, 512, 512, 3), np.uint8); B = np.zeros((N, 512, 512, 3), np.uint8); H = np.zeros((N, 512, 512), bool); L = np.zeros((N, 128, 128), np.uint8)
for i in range(N):
    b, known, same = bases[rng.integers(len(bases))]; h = sample_hole(known)
    W[i] = wm.embed(b, rng.integers(0, 2, wm.nbits)); B[i] = b; H[i] = h
    lab = ~h
    if known is not None and same: lab = lab | (h & known)          # the base's own real watermark shows through the hole
    L[i] = downsample_mask(lab.astype(np.uint8))
    if i % 200 == 0: print(f"   embed {i}/{N}  {time.time() - t0:.0f}s", flush=True)
comp = lambda Wm: np.where(H[..., None], B, Wm)
data_v = {}
if "none" in VARIANTS: data_v["none"] = comp(W)
if "post" in VARIANTS or "pre" in VARIANTS:
    from diffusers import AutoencoderKL
    from safetensors.torch import load_file
    d = "external/vae/flux/vae"; vae = AutoencoderKL.from_config(AutoencoderKL.load_config(d)); vae.load_state_dict(load_file(f"{d}/diffusion_pytorch_model.safetensors"))
    vae = vae.cuda().eval().half()

    @torch.no_grad()
    def rt(X):
        out = np.zeros_like(X)
        for i in range(0, len(X), 16):
            x = torch.from_numpy(X[i:i + 16]).cuda().permute(0, 3, 1, 2).half() / 127.5 - 1
            y = vae.decode(vae.encode(x).latent_dist.mode()).sample
            out[i:i + 16] = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        return out
    if "post" in VARIANTS: data_v["post"] = rt(comp(W))
    if "pre" in VARIANTS: data_v["pre"] = comp(rt(W))
    del vae; torch.cuda.empty_cache()
del W
print(f"generated {N} x {list(data_v)}  hole area median {np.median(1 - L.reshape(N, -1).mean(1)):.3f}  {time.time() - t0:.0f}s", flush=True)

# ---- real train images of this method, evaluation helpers
tr = data.load_split("train", [method]); names = [tr[i]["image_name"] for i in range(len(tr))]; Xr = [tr[i]["image"] for i in range(len(tr))]
Mr = [g128[n] for n in names]; fold = df.loc[names].fold.values; isfull = df.loc[names].full_watermark.values
fi = np.nonzero(isfull)[0]; G = np.stack([Mr[i] for i in fi]).astype(bool); ff = fold[fi]


def zero_shot(Pf):
    return float(np.median([auc(-Pf[i].ravel(), ~G[i].ravel()) for i in range(len(fi))])), float(max(miou_batch(Pf > t, G).mean() for t in TS))


def cv_full(Pf):
    out = np.zeros(len(fi))
    for k in range(5):
        trn = ff != k; t = max(TS, key=lambda t: miou_batch(Pf[trn] > t, G[trn]).mean()); out[~trn] = miou_batch(Pf[~trn] > t, G[~trn])
    return float(out.mean())


def existing(tag):
    try:
        nm, P = load_maps(tag, "train"); P = dict(zip(nm, P)); return np.stack([P[names[i]] for i in fi])
    except Exception:
        return None


ENS = {"pixelseal": "ens3_pixelseal", "trustmark": "ens2_trustmark", "stegastamp": "ens2_stegastamp"}[method]
print("reference (existing maps, same real full images):")
for tag in (f"e2eR512pre_{method}", f"e2efluxpre_{method}", f"e2eFHpre_{method}"):
    P = existing(tag)
    if P is not None: a, mi = zero_shot(P); print(f"   zero-shot  {tag:28s} hole AUC {a:.3f} | mIoU@best {mi:.3f}")
for tag in (f"e2eR512_{method}", f"e2eFH_{method}", ENS):
    P = existing(tag)
    if P is not None: print(f"   fine-tuned {tag:28s} 5-fold full CV {cv_full(P):.4f}")
Pens = existing(ENS)

# ---- train / evaluate each variant
for v, X in data_v.items():
    # FIX 02:35: e2e.Net shares the decoder's layers with the wrapper, so every variant needs a FRESH decoder.
    # Logs written before this fix: only the FIRST variant of a TrustMark / PixelSeal run is clean; later variants
    # started from a backbone already fine-tuned on real folds and are inflated (StegaStamp is not affected).
    wmv = load_watermark(method, models_dir=data.TASK_DIR / "models", device="cuda")
    net = e2e.Net(wmv).cuda().float(); net.res = 512
    e2e.train(net, X, L, STEPS, 1e-3, 16, 0)
    pre = e2e.predict(net, [Xr[i] for i in fi]); a, mi = zero_shot(pre)
    print(f"{method} [{v}] zero-shot: hole AUC {a:.3f} | mIoU@best {mi:.3f}   {time.time() - t0:.0f}s", flush=True)
    base = copy.deepcopy(net.state_dict()); held = np.zeros((len(fi), 128, 128), np.float32)
    for k in range(5):
        net.load_state_dict(base); idx = fi[ff != k]
        e2e.train(net, [Xr[i] for i in idx], [Mr[i] for i in idx], max(20, 30 * len(idx) // 16), 3e-4, 16, k + 1)
        held[ff == k] = e2e.predict(net, [Xr[i] for i in fi[ff == k]])
    msg = f"{method} [{v}] fine-tuned on real full: 5-fold full CV {cv_full(held):.4f}"
    if Pens is not None: msg += f" | mean({ENS}, new) {cv_full((Pens + held) / 2):.4f}"
    print(msg + f"   {time.time() - t0:.0f}s", flush=True)
    if not DRY:
        try:                                                        # the project's inode quota may be exhausted
            np.savez_compressed(f"{OUTDIR}/realsynth_{v}_{method}_train.npz", names=np.array([names[i] for i in fi]), prob=held.astype(np.float16), pre=pre.astype(np.float16))
        except OSError as e:
            print("   maps not saved:", e, flush=True)
    del net; torch.cuda.empty_cache()
print("done", f"{time.time() - t0:.0f}s")
