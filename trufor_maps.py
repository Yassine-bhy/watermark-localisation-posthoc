"""Step 2: TruFor (pretrained forgery localizer, GRIP-UNINA; code + weights in external/trufor) maps for train images.
    python trufor_maps.py [--only_full 1] [--split train]
Output: results/trufor_<split>.npz  names, prob (128 x 128 'manipulated' probability), conf, det (image score)."""
import argparse, sys, time
import numpy as np, torch, torch.nn.functional as F
import data

SRC = data.HERE / "external" / "trufor" / "code" / "test_docker" / "src"
sys.path.insert(0, str(SRC))
from config import _C as config  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="train"); ap.add_argument("--only_full", type=int, default=1)
ap.add_argument("--device", default="cpu"); a = ap.parse_args()
config.defrost(); config.merge_from_file(str(SRC / "trufor.yaml")); config.freeze()
from models.cmx.builder_np_conf import myEncoderDecoder  # noqa: E402
torch.set_num_threads(48)
model = myEncoderDecoder(cfg=config)
ck = torch.load(str(data.HERE / "external" / "trufor" / "weights" / "weights" / "trufor.pth.tar"), map_location="cpu", weights_only=False)
model.load_state_dict(ck["state_dict"]); model = model.to(a.device).eval()
ds = data.load_split(a.split); df = data.train_table().set_index("name") if a.split == "train" else None
names, probs, confs, dets = [], [], [], []; t0 = time.time()
with torch.no_grad():
    for i in range(len(ds)):
        n = ds.meta[i]["image_name"]
        if a.only_full and df is not None and not df.loc[n].full_watermark:
            continue
        it = ds[i]; x = torch.from_numpy(it["image"].copy()).permute(2, 0, 1)[None].float().div(255).to(a.device)
        pred, conf, det, npp = model(x)
        p = F.softmax(pred, 1)[:, 1:2]; c = torch.sigmoid(conf)
        names.append(n); probs.append(F.adaptive_avg_pool2d(p, 128)[0, 0].cpu().numpy()); confs.append(F.adaptive_avg_pool2d(c, 128)[0, 0].cpu().numpy())
        dets.append(float(torch.sigmoid(det))); 
        if len(names) % 25 == 0:
            print(len(names), f"{time.time() - t0:.0f}s", flush=True)
np.savez_compressed(data.RESULTS / f"trufor_{a.split}.npz", names=np.array(names), prob=np.stack(probs).astype(np.float16), conf=np.stack(confs).astype(np.float16), det=np.array(dets))
print("done", len(names), f"{time.time() - t0:.0f}s")
