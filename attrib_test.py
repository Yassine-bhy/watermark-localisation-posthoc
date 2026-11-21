"""Quick check: within-image AUC of decoder attribution maps (occlusion, gradient) on 8 full + 8 partial train images per method."""
import json, time
import numpy as np, torch, data
from attrib import occlusion, gradient
from diagnostics import auc
from wmloc.watermarks import load_watermark
df = data.train_table().set_index('name'); g = data.train_masks128(); out = {}
for m in data.METHODS:
    wm = load_watermark(m, models_dir=data.TASK_DIR / 'models', device='cuda'); ds = data.load_split('train', [m])
    for reg in ('full', 'partial'):
        idx = [i for i, mm in enumerate(ds.meta) if df.loc[mm['image_name']].regime == reg][:8]; R = []; t0 = time.time()
        for i in idx:
            it = ds[i]; x = torch.from_numpy(it['image']).permute(2, 0, 1)[None].float().div(255).cuda()
            s = torch.from_numpy(it['message'].astype(np.float32) * 2 - 1).cuda(); G = g[it['image_name']].ravel().astype(bool)
            oc = occlusion(wm, x, s).cpu().numpy().ravel()
            gr = gradient(wm, x, s).cpu().numpy().ravel() if m != 'stegastamp' else None
            R.append((auc(oc, G), auc(gr, G) if gr is not None else float('nan')))
        R = np.array(R); out[f'{m}_{reg}'] = R.tolist()
        print(f"{m:10s} {reg:7s} within-image AUC occlusion {np.nanmedian(R[:,0]):.3f} {np.round(R[:,0],2)} | gradient {np.nanmedian(R[:,1]):.3f} {np.round(R[:,1],2)} ({(time.time()-t0)/len(idx):.1f}s/img)", flush=True)
json.dump(out, open(data.RESULTS / 'attrib_test.json', 'w'))
