"""Simulator search (REPORT_FULL.md, idea 2): which way of weakening the watermark makes real-domain synthetic full
images transfer best to the real ones?  Same data recipe and protocol as test_realsynth.py; only the processing varies.
    python research/tests/test_realsynth_sim.py <method> [N=1000] [STEPS=2500] [variants=pre,pre2,preN05,preN15,prepost]

W = watermarked base (whole frame), b = base, H = hole.  rt = FLUX VAE encode/decode.
  none     composite(W, b, H)
  post     rt(composite(W, b, H))
  pre      composite(rt(W), b, H)              <- best so far (TrustMark zero-shot mIoU 0.638)
  pre2     composite(rt(rt(W)), b, H)          two round trips of the watermarked image
  preNxx   composite(rt_noise(W, xx %), b, H)  Gaussian noise of xx % of the latent's std added before decoding
  prepost  rt(composite(rt(W), b, H))          pre, then one more round trip of everything
  preSD / preSDXL   pre with the SD / SDXL VAE of external/vae instead of FLUX (if loadable)
  mix:a+b+c         every image drawn at random from recipes a, b, c (e.g. mix:none+post+pre): the network cannot
                    rely on "round-tripped vs untouched" alone, which is what broke PixelSeal with pure 'pre'
Score = zero-shot on the real FULL train images of <method> (hole AUC, mIoU at the best global threshold, and the
fold-wise CV version), then the usual fine-tune on real full images (5 folds) and mean with the current ensemble.
Decision rule (written before the first run): a candidate replaces 'pre' only if its zero-shot mIoU is >= pre + 0.015
on TrustMark and it does not lose on a second method.  'pre' is run first as a replicate (noise estimate).
Writes research/out/realsim_<variant>_<method>_train.npz (prob = held-out fine-tuned maps, pre = zero-shot maps)."""
import copy, os, sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch
import data, e2e
from diagnostics import auc, best_gate, miou_batch
from evalmaps import load_maps
from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark
from diffusers import AutoencoderKL
from safetensors.torch import load_file

method = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
VARIANTS = (sys.argv[4] if len(sys.argv) > 4 else "pre,pre2,preN05,preN15,prepost").split(",")
OUTDIR = os.environ.get("RS_OUT", "research/out"); DRY = bool(os.environ.get("RS_DRY"))
t0 = time.time(); rng = np.random.default_rng(12345 + data.METHODS.index(method)); torch.manual_seed(0)
e2e.OUT = 64; e2e.DEEP = False
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
print(f"{method}: bases {len(bases)}, hole masks {len(holes)}, variants {VARIANTS}  {time.time() - t0:.0f}s", flush=True)


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
print(f"embedded {N}  {time.time() - t0:.0f}s", flush=True)
comp = lambda A: np.where(H[..., None], B, A)
VAES = {}


def vae_of(name):
    if name not in VAES:
        d = {"flux": "external/vae/flux/vae", "sdxl": "external/vae/sdxl", "sd": "external/vae/sd"}[name]
        d = d + "/vae" if os.path.isdir(d + "/vae") else d
        v = AutoencoderKL.from_config(AutoencoderKL.load_config(d)); f = [x for x in os.listdir(d) if x.endswith(".safetensors")][0]
        v.load_state_dict(load_file(f"{d}/{f}")); VAES[name] = v.cuda().eval().half()
    return VAES[name]


@torch.no_grad()
def rt(A, noise=0.0, name="flux"):
    vae = vae_of(name); out = np.zeros_like(A); gen = torch.Generator("cuda").manual_seed(1)
    for i in range(0, len(A), 16):
        x = torch.from_numpy(A[i:i + 16]).cuda().permute(0, 3, 1, 2).half() / 127.5 - 1
        z = vae.encode(x).latent_dist.mode()
        if noise:
            z = z + noise * z.float().std() * torch.randn(z.shape, generator=gen, device="cuda", dtype=z.dtype)
        y = vae.decode(z).sample
        out[i:i + 16] = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return out


CACHE = {}


def rtW():
    if "rtW" not in CACHE: CACHE["rtW"] = rt(W)
    return CACHE["rtW"]


def make(v):
    if v == "none": return comp(W)
    if v == "post": return rt(comp(W))
    if v == "pre": return comp(rtW())
    if v == "pre2": return comp(rt(rtW()))
    if v == "prepost": return rt(comp(rtW()))
    if v.startswith("preN"): return comp(rt(W, noise=float(v[4:]) / 100))
    if v == "preSD": return comp(rt(W, name="sd"))
    if v == "preSDXL": return comp(rt(W, name="sdxl"))
    if v == "post2": return rt(rt(comp(W)))                    # stronger weakening, still no hole / surroundings contrast
    if v == "postSD": return rt(comp(W), name="sd")
    if v == "postSDXL": return rt(comp(W), name="sdxl")
    if v.startswith("postN"): return rt(comp(W), noise=float(v[5:]) / 100)
    if v.startswith("postJ"):                                  # post, then JPEG at quality xx
        import io
        from PIL import Image
        A = rt(comp(W)); q = int(v[5:])
        for j in range(len(A)):
            buf = io.BytesIO(); Image.fromarray(A[j]).save(buf, "JPEG", quality=q); A[j] = np.asarray(Image.open(buf).convert("RGB"))
        return A
    # ---- candidates without a VAE; all of them process hole and surroundings alike (no shortcut)
    if v.startswith("str"):                                    # weaker fresh watermark: b + xx % of (W - b)
        a = float(v[3:]) / 100
        return comp(np.clip(np.rint(B.astype(np.float32) + a * (W.astype(np.float32) - B.astype(np.float32))), 0, 255).astype(np.uint8))
    if v.startswith("blur"):                                   # Gaussian blur of the composite, sigma = xx / 10 px
        from scipy import ndimage as ndi
        s = float(v[4:]) / 10; A = comp(W)
        for j in range(N): A[j] = ndi.gaussian_filter(A[j], (s, s, 0))
        return A
    if v.startswith("du"):                                     # composite resized to xx px and back (bicubic)
        from PIL import Image
        r = int(v[2:]); A = comp(W)
        for j in range(N): A[j] = np.asarray(Image.fromarray(A[j]).resize((r, r), Image.BICUBIC).resize((512, 512), Image.BICUBIC))
        return A
    if v.startswith("jpg"):                                    # composite saved as JPEG quality xx
        import io
        from PIL import Image
        q = int(v[3:]); A = comp(W)
        for j in range(N):
            buf = io.BytesIO(); Image.fromarray(A[j]).save(buf, "JPEG", quality=q); A[j] = np.asarray(Image.open(buf).convert("RGB"))
        return A
    if v.startswith("hires"):                                  # watermark embedded on the base upscaled to xx px, composite there, downscale
        from PIL import Image
        r = int(v[5:]) if len(v) > 5 else 1024; A = np.zeros_like(W)
        for j in range(N):
            bu = np.asarray(Image.fromarray(B[j]).resize((r, r), Image.BICUBIC)); wu = wm.embed(bu, MSG[j])
            hu = np.asarray(Image.fromarray(H[j].astype(np.uint8) * 255).resize((r, r), Image.NEAREST)) > 127
            A[j] = np.asarray(Image.fromarray(np.where(hu[..., None], bu, wu)).resize((512, 512), Image.LANCZOS))
        return A
    if v.startswith("mix:"):                                   # each image drawn from one of the listed recipes
        parts = v[4:].split("+"); pick = np.random.default_rng(7).integers(0, len(parts), N); out = make(parts[0])
        for j in range(1, len(parts)):
            a = make(parts[j]); out[pick == j] = a[pick == j]; del a
        return out
    raise ValueError(v)


tr = data.load_split("train", [method]); names = [tr[i]["image_name"] for i in range(len(tr))]; Xr = [tr[i]["image"] for i in range(len(tr))]
Mr = [g128[n] for n in names]; fold = df.loc[names].fold.values; isfull = df.loc[names].full_watermark.values
fi = np.nonzero(isfull)[0]; G = np.stack([Mr[i] for i in fi]).astype(bool); ff = fold[fi]


def cv_full(Pf):
    out = np.zeros(len(fi))
    for k in range(5):
        trn = ff != k; t = max(TS, key=lambda t: miou_batch(Pf[trn] > t, G[trn]).mean()); out[~trn] = miou_batch(Pf[~trn] > t, G[~trn])
    return float(out.mean())


ENS = {"pixelseal": "ens3_pixelseal", "trustmark": "ens2_trustmark", "stegastamp": "ens2_stegastamp"}[method]
nm, P = load_maps(ENS, "train"); P = dict(zip(nm, P)); Pens = np.stack([P[names[i]] for i in fi]); print(f"reference {ENS}: full CV {cv_full(Pens):.4f}", flush=True)
MADE = {}
for spec in VARIANTS:                                              # "recipe" or "recipe@steps" (training length)
    v, st = (spec.split("@") + [STEPS])[:2]; st = int(st)
    try:
        if v not in MADE: MADE[v] = make(v)
        X = MADE[v]
    except Exception as e:
        print(f"{method} [{spec}] skipped: {type(e).__name__}: {str(e)[:120]}", flush=True); continue
    dmean = float(np.mean([np.abs(X[j].astype(np.int16) - W[j].astype(np.int16))[~H[j]].mean() for j in range(0, N, max(1, N // 100))]))   # change of the watermarked area
    # FIX 02:35: e2e.Net shares the decoder's layers with the wrapper, so every variant needs a FRESH decoder.
    # Before this fix each variant started from the previous variant's backbone (already fine-tuned on real folds):
    # all TrustMark / PixelSeal numbers of non-first variants in logs written before 02:35 are inflated.
    wmv = load_watermark(method, models_dir=data.TASK_DIR / "models", device="cuda")
    net = e2e.Net(wmv).cuda().float(); net.res = 512
    e2e.train(net, X, L, st, 1e-3, 16, 0); v = f"{v}@{st}"
    pre = e2e.predict(net, [Xr[i] for i in fi])
    a = float(np.median([auc(-pre[i].ravel(), ~G[i].ravel()) for i in range(len(fi))])); mi = float(max(miou_batch(pre > t, G).mean() for t in TS))
    base = copy.deepcopy(net.state_dict()); held = np.zeros((len(fi), 128, 128), np.float32)
    for k in range(5):
        net.load_state_dict(base); idx = fi[ff != k]
        e2e.train(net, [Xr[i] for i in idx], [Mr[i] for i in idx], 20 if DRY else max(20, 30 * len(idx) // 16), 3e-4, 16, k + 1)
        held[ff == k] = e2e.predict(net, [Xr[i] for i in fi[ff == k]])
    print(f"{method} [{v:22s}] change of watermarked area {dmean:.2f}/255 | zero-shot: hole AUC {a:.3f}, mIoU@best {mi:.3f}, CV {cv_full(pre):.4f} | fine-tuned CV {cv_full(held):.4f} | "
          f"mean({ENS}, ft) {cv_full((Pens + held) / 2):.4f} | mean({ENS}, ft, zero-shot) {cv_full((Pens + held + pre) / 3):.4f}   {time.time() - t0:.0f}s", flush=True)
    if not DRY:
        try:
            np.savez_compressed(f"{OUTDIR}/realsim_{v.replace(':', '').replace('+', '-').replace('@', '_s')}_{method}_train.npz", names=np.array([names[i] for i in fi]), prob=held.astype(np.float16), pre=pre.astype(np.float16))
        except OSError as e:
            print("   maps not saved:", e, flush=True)
    del net, wmv; torch.cuda.empty_cache()
print("done", f"{time.time() - t0:.0f}s")
