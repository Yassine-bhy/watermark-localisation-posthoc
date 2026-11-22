"""Kill test (1 GPU, ~8 min): counterfactual re-embedding + RISE-style regression on FULL images (holes).
Fills: A = Emb(y, not m) (complementary message), B = Emb(y, m) (same message again).
For K random 16x16 binary masks M (upsampled, nearest): x = M*y + (1-M)*fill; s = mean_k (2m_k-1) * logit_k(x).
Ridge regression s ~ M gives one coefficient per cell = how much keeping the ORIGINAL pixels of that cell helps.
Expected if it works: watermarked cells -> large positive coefficient with fill A (original carries m, fill carries
not-m); clean holes -> smaller with A and NEGATIVE with fill B (fill adds m where there was none).
Reports first whether Dec(Emb(y, not m)) actually flips (sanity), then the within-image AUC (hole = positive) of
-coefA, -coefB and coefB-coefA style maps, next to our learned map, on up to N full images with hole >= 1 %.
Pass threshold (set in advance): median AUC >= 0.80 for a method (our learned maps are at 0.90-0.93; below 0.80 a new
map is not worth fusing)."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F, data
from diagnostics import auc
from evalmaps import load_maps
from wmloc.watermarks import load_watermark
TAGS = {"pixelseal": "ens3_pixelseal", "trustmark": "ens2_trustmark"}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15; K = 1536; GRID = 16
df = data.train_table().set_index("name"); T = lambda a: torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()
for meth, tag in TAGS.items():
    wm = load_watermark(meth, models_dir=data.TASK_DIR / "models", device="cuda"); names, P = load_maps(tag, "train"); P = dict(zip(names, P))
    ds = data.load_split("train", [meth]); res = {k: [] for k in ("ours", "A", "B", "B-A")}; san = []; n = 0; rng = np.random.default_rng(0)
    for i in range(len(ds)):
        it = ds[i]; nm = it["image_name"]; gt = it["mask128"].astype(bool)
        if not df.loc[nm].full_watermark or (~gt).mean() < 0.01: continue
        y = it["image"]; m = it["message"]; s = torch.from_numpy(m.astype(np.float32) * 2 - 1).cuda()
        fa, fb = T(wm.embed(y, 1 - m)), T(wm.embed(y, m)); ty = T(y)
        sc = lambda x: (wm.logits(x).to(s.device) * s[None]).mean(1)
        san.append([float(sc(ty)), float(sc(fa)), float(sc(fb))])
        M = (torch.from_numpy(rng.random((K, 1, GRID, GRID)) < 0.5)).float().cuda(); coef = {}
        for key, fill in (("A", fa), ("B", fb)):
            out = []
            for j in range(0, K, 64):
                mm = F.interpolate(M[j:j + 64], size=512, mode="nearest"); out.append(sc(mm * ty + (1 - mm) * fill))
            v = torch.cat(out); X = M.flatten(1) - 0.5; v = v - v.mean()
            w = torch.linalg.solve(X.T @ X + 1.0 * torch.eye(GRID * GRID, device=X.device), X.T @ v)
            coef[key] = F.interpolate(w.view(1, 1, GRID, GRID), size=128, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        hole = ~gt.ravel()
        res["ours"].append(auc(-P[nm].ravel(), hole)); res["A"].append(auc(-coef["A"].ravel(), hole)); res["B"].append(auc(-coef["B"].ravel(), hole))
        res["B-A"].append(auc(-(coef["A"] + coef["B"]).ravel(), hole))
        n += 1
        if n == N: break
    san = np.array(san)
    print(f"{meth}: n={n}  known-message score: original {np.median(san[:,0]):.2f} | Emb(y, not m) {np.median(san[:,1]):.2f} | Emb(y, m) {np.median(san[:,2]):.2f}")
    print("   median within-image hole AUC:", {k: round(float(np.median(v)), 3) for k, v in res.items()}, flush=True)
