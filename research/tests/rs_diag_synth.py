"""Diagnostic: is the remaining loss on FULL images a data-fidelity problem or a model problem?
    [RS_OUTGRID=64|128] [RS_DEEP=1] python research/tests/rs_diag_synth.py <method> <recipe> <steps> [N=3000]

Train an RS full-image model exactly as realsynth_prod.py does, on N images, and score it on
  (a) 400 HELD-OUT synthetic RS images (same generator, never trained on), and
  (b) the real full train images of <method> (zero-shot),
both split by hole size (< 3 %, 3-10 %, > 10 % of the image), with one global threshold per set.
Reading:  synthetic small holes fine but real ones missed  -> the simulator is off (fidelity is the lever);
          small holes missed on synthetic too              -> the model cannot resolve them (resolution / capacity
                                                              is the lever), whatever the data.
Decision rule (written before running): if the synthetic score on holes < 3 % is below 0.65, test a finer output grid
(RS_OUTGRID=128) and keep it only if it raises that number by >= 0.03 without lowering the real score.
Writes nothing but its stdout."""
import os, sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch
import data, e2e
from diagnostics import auc, best_gate, miou_batch
from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark
from diffusers import AutoencoderKL
from safetensors.torch import load_file

method, RECIPE, STEPS = sys.argv[1], sys.argv[2], int(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
HELD = 400; NT = N + HELD; t0 = time.time(); rng = np.random.default_rng(999 + data.METHODS.index(method)); torch.manual_seed(0)
e2e.OUT = int(os.environ.get("RS_OUTGRID", 64)); e2e.DEEP = bool(os.environ.get("RS_DEEP"))
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
for m in data.METHODS:
    ztr = np.load(data.RESULTS / "signals" / f"{m}_train.npz"); zte = np.load(data.RESULTS / "signals" / f"{m}_test.npz")
    gate = best_gate(ztr["global_soft"], df.loc[ztr["names"].tolist()].full_watermark.values)
    part = set(np.array(zte["names"])[zte["global_soft"] < gate].tolist()); te = data.load_split("test", [m])
    for i in range(len(te)):
        if te.meta[i]["image_name"] in part:
            bases.append((te[i]["image"], None, m == method))
# held-out synthetic images use bases and hole masks the training images never see
nb = len(bases); perm = rng.permutation(nb); b_te, b_tr = perm[:nb // 8], perm[nb // 8:]
hp = rng.permutation(len(holes)); h_te, h_tr = hp[:len(holes) // 6], hp[len(holes) // 6:]
print(f"{method} recipe {RECIPE} steps {STEPS} N {N} out grid {e2e.OUT} deep {e2e.DEEP}: bases {len(b_tr)} train / {len(b_te)} held-out, hole masks {len(h_tr)} / {len(h_te)}  {time.time() - t0:.0f}s", flush=True)


def aug(h):
    if rng.random() < 0.5: h = h[:, ::-1]
    if rng.random() < 0.5: h = h[::-1]
    if rng.random() < 0.5: h = h.T
    return h


def sample_hole(known, pool):
    for _ in range(8):
        h = aug(holes[pool[rng.integers(len(pool))]])
        if rng.random() < 0.25:
            h2 = h | aug(holes[pool[rng.integers(len(pool))]])
            if h2.mean() < 0.4: h = h2
        if h.mean() > 0.5: continue
        if known is None or not (h & known).any(): break
    return np.ascontiguousarray(h)


wm = load_watermark(method, models_dir=data.TASK_DIR / "models", device="cuda")
X = np.zeros((NT, 512, 512, 3), np.uint8); L = np.zeros((NT, 128, 128), np.uint8); H = np.zeros((NT, 512, 512), bool); B = np.zeros((NT, 512, 512, 3), np.uint8)
for i in range(NT):
    bp, hpool = (b_tr, h_tr) if i < N else (b_te, h_te)
    b, known, same = bases[bp[rng.integers(len(bp))]]; h = sample_hole(known, hpool)
    X[i] = wm.embed(b, rng.integers(0, 2, wm.nbits)); B[i] = b; H[i] = h
    lab = ~h
    if known is not None and same: lab = lab | (h & known)
    L[i] = downsample_mask(lab.astype(np.uint8))
np.copyto(X, B, where=H[..., None]); del B, H
parts = RECIPE[4:].split("+") if RECIPE.startswith("mix:") else [RECIPE]; pick = np.random.default_rng(7).integers(0, len(parts), NT)
if "post" in parts:
    d = "external/vae/flux/vae"; vae = AutoencoderKL.from_config(AutoencoderKL.load_config(d)); vae.load_state_dict(load_file(f"{d}/diffusion_pytorch_model.safetensors")); vae = vae.cuda().eval().half()
    ids = np.nonzero(pick == parts.index("post"))[0]
    with torch.no_grad():
        for j in range(0, len(ids), 16):
            k = ids[j:j + 16]; x = torch.from_numpy(X[k]).cuda().permute(0, 3, 1, 2).half() / 127.5 - 1
            y = vae.decode(vae.encode(x).latent_dist.mode()).sample
            X[k] = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    del vae; torch.cuda.empty_cache()
print(f"generated {NT}  {time.time() - t0:.0f}s", flush=True)

net = e2e.Net(wm).cuda().float(); net.res = 512
e2e.train(net, X[:N], L[:N], STEPS, 1e-3, 16, 0)
tr = data.load_split("train", [method]); idx = [i for i in range(len(tr)) if tr.meta[i].get("full_watermark")]
Xr = [tr[i]["image"] for i in idx]; Gr = np.stack([g128[tr.meta[i]["image_name"]] for i in idx]).astype(bool)
Ps = e2e.predict(net, [X[i] for i in range(N, NT)]); Gs = L[N:].astype(bool); Pr = e2e.predict(net, Xr)


def table(P, G, label):
    hf = 1 - G.reshape(len(G), -1).mean(1); ok = (hf > 0) & (hf < 1); P, G, hf = P[ok], G[ok], hf[ok]
    t = max(TS, key=lambda t: miou_batch(P > t, G).mean()); s = miou_batch(P > t, G)
    a = np.array([auc(-P[i].ravel(), ~G[i].ravel()) for i in range(len(G))])
    missed = np.array([not ((P[i] <= t) & ~G[i]).any() for i in range(len(G))])
    row = f"{label:22s} n={len(G):3d}  mIoU {s.mean():.3f} (thr {t:.2f}) | hole AUC {np.median(a):.3f} | hole missed entirely {missed.mean():.2f}"
    for name, lo, hi in (("<3%", 0, .03), ("3-10%", .03, .1), (">10%", .1, 1)):
        k = (hf >= lo) & (hf < hi)
        row += f" | {name}: n={int(k.sum())} mIoU {s[k].mean():.3f} AUC {np.median(a[k]):.3f} missed {missed[k].mean():.2f}" if k.any() else f" | {name}: n=0"
    print(row, flush=True)


table(Ps, Gs, "held-out SYNTHETIC")
table(Pr, Gr, "REAL full (zero-shot)")
print("done", f"{time.time() - t0:.0f}s")
