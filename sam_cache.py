"""SAM (ViT-B) prompted at the peak of our map, for images the decode gate may call partial (global_soft <= 0.6).
research/tests/test_samprompt.py logic; cached so the CV / submission can use it without a GPU.
    python sam_cache.py --tags pixelseal=ens3_pixelseal,trustmark=ens2_trustmark,stegastamp=ens2_stegastamp
Writes results/samcache_<method>_<split>.npz: names, masks (N, 3, 128, 128) bool, point (N, 2) [y, x] on the 128 grid."""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
from scipy import ndimage as ndi
from segment_anything import SamPredictor, sam_model_registry
import data
from evalmaps import load_maps

ap = argparse.ArgumentParser(); ap.add_argument("--tags", required=True); ap.add_argument("--device", default="cpu"); ap.add_argument("--splits", default="train,test"); a = ap.parse_args()
torch.set_num_threads(48)
sam = sam_model_registry["vit_b"](checkpoint=str(data.HERE / "external" / "sam" / "sam_vit_b_01ec64.pth")).to(a.device).eval(); pred = SamPredictor(sam)
down = lambda m: (F.avg_pool2d(torch.from_numpy(m.astype(np.float32))[None, None], 4)[0, 0] > 0.5).numpy()
t0 = time.time()
for kv in a.tags.split(","):
    meth, tag = kv.split("=")
    for split in a.splits.split(","):
        names, P = load_maps(tag, split); P = dict(zip(names, P))
        gs = dict(zip(*[np.load(data.RESULTS / "signals" / f"{meth}_{split}.npz")[k].tolist() for k in ("names", "global_soft")]))
        ds = data.load_split(split, [meth]); out_n, out_m, out_p = [], [], []
        for i, mt in enumerate(ds.meta):
            n = mt["image_name"]
            if gs[n] > 0.6:
                continue
            q = ndi.gaussian_filter(P[n], 1.5); y, x = np.unravel_index(np.argmax(q), q.shape)
            pred.set_image(ds[i]["image"])
            with torch.no_grad():
                ms, _, _ = pred.predict(point_coords=np.array([[x * 4 + 2, y * 4 + 2]]), point_labels=np.array([1]), multimask_output=True)
            out_n.append(n); out_m.append(np.stack([down(k) for k in ms])); out_p.append((y, x))
        for t in range(10):                                  # intermittent "Disk quota exceeded" on this filesystem
            try:
                np.savez_compressed(data.RESULTS / f"samcache_{meth}_{split}.npz", names=np.array(out_n), masks=np.array(out_m), point=np.array(out_p)); break
            except OSError as e:
                print("save failed, retry", t, e, flush=True); time.sleep(30)
        print(meth, split, len(out_n), f"{time.time() - t0:.0f}s", flush=True)
