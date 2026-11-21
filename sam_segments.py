"""Idea 3: SAM automatic segments for every image (to snap blurry maps to object / region boundaries).
    python sam_segments.py
SAM ViT-B automatic mask generator on the 512 x 512 image; each segment stored at 128 x 128 (the score grid).
Writes results/sam_{train,test}.npz: names, masks (all segments concatenated, packed bits), offsets."""
import time
import numpy as np, torch, torch.nn.functional as F
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import data

sam = sam_model_registry["vit_b"](checkpoint=str(data.HERE / "external" / "sam" / "sam_vit_b_01ec64.pth")).cuda().eval()
gen = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.8, stability_score_thresh=0.85, min_mask_region_area=64)
t0 = time.time()
for split in ("train", "test"):
    ds = data.load_split(split); names, packs, offs = [], [], [0]
    for i in range(len(ds)):
        it = ds[i]; names.append(it["image_name"])
        with torch.no_grad():
            ms = gen.generate(it["image"])
        if ms:
            m = torch.from_numpy(np.stack([x["segmentation"] for x in ms])).float()[:, None]
            m = (F.avg_pool2d(m, 4)[:, 0] > 0.5).numpy()
            m = m[m.reshape(len(m), -1).any(1)]
        else:
            m = np.zeros((0, 128, 128), bool)
        packs.append(np.packbits(m.reshape(len(m), -1), axis=1)); offs.append(offs[-1] + len(m))
        if i % 100 == 0:
            print(split, i, len(m), f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(data.RESULTS / f"sam_{split}.npz", names=np.array(names), masks=np.concatenate(packs), offsets=np.array(offs))
    print(split, "done", f"{time.time() - t0:.0f}s", flush=True)
