"""Builds notebooks/02_network.ipynb (step 2: WAM-based segmentation network). Only loads saved results."""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 02 — Segmentation network (Watermark Anything backbone)
Model: WAM's pretrained extractor (SAM ViT-B + pixel decoder, MIT weights), new head = per method one
"watermarked?" map + one map per message bit. Pretrained on synthetic edits (COCO + the kit's embedders), then
fine-tuned on the real train images with 5-fold CV. **Scores: official `miou_image`; the CV numbers come from
held-out folds only.** The *agreement* map = mean over bits of (2m−1)·tanh(predicted bit logit) with the known message.""")

code("""import sys, json; sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import data
from metrics import miou_image
pd.set_option('display.width', 160)
R = data.RESULTS; df = data.train_table().set_index('name'); gts = data.train_masks128()
ds = data.load_split('train'); idx = {m['image_name']: i for i, m in enumerate(ds.meta)}
def jl(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []
from pathlib import Path""")

md("## 1. Synthetic training data (first shard of each method): image and label (white = watermarked)")
code("""fig, axs = plt.subplots(6, 8, figsize=(20, 15))
for r, m in enumerate(data.METHODS):
    d = R / 'synth' / f'{m}_0'; I = np.load(d / 'images.npy', mmap_mode='r'); M = np.load(d / 'masks.npy')
    meta = json.loads((d / 'meta.json').read_text())['samples']
    for j in range(8):
        axs[2*r, j].imshow(I[j]); axs[2*r, j].set_title(f"{m} · {meta[j]['kind']}", fontsize=9)
        axs[2*r+1, j].imshow(M[j], cmap='gray', vmin=0, vmax=1)
for a in axs.ravel(): a.axis('off')
plt.tight_layout(); plt.show()
kinds = pd.DataFrame([dict(method=json.loads((d / 'meta.json').read_text())['method'], **s) for d in sorted((R / 'synth').glob('*_*'))
                      for s in json.loads((d / 'meta.json').read_text())['samples']])
display(pd.crosstab(kinds.method, kinds.kind, margins=True))""")

md("## 2. Pretraining (synthetic only): loss, and score on the 510 real train images (never trained on)")
code("""L = jl(R / 'log_wam256_pretrain.jsonl')
loss = pd.DataFrame([l for l in L if 'loss' in l]); ev = [l for l in L if 'real_train' in l]
fig, ax = plt.subplots(1, 2, figsize=(14, 3.5))
ax[0].plot(loss.step, loss.loss); ax[0].set_xlabel('step'); ax[0].set_title('training loss (synthetic)')
if ev:
    e = pd.DataFrame([{'step': l['step'], **{m: l['real_train'][m]['miou_at_05'] for m in data.METHODS}} for l in ev]).set_index('step')
    e.plot(ax=ax[1], marker='o'); ax[1].set_title('real train mIoU at threshold 0.5'); ax[1].axhline(0.473, ls='--', c='gray', label='step 1 CV')
    ax[1].legend()
plt.show()
if ev: display(pd.DataFrame(ev[-1]['real_train']).T)""")

md("## 3. Fine-tuning, 5-fold CV (held-out folds)")
code("""rows = []
for k in range(5):
    for l in jl(R / f'log_wam256_finetune_f{k}.jsonl'):
        if 'heldout' in l:
            rows.append({'fold': k, **{m: l['heldout'][m]['miou_at_05'] for m in data.METHODS}, 'all': l['heldout']['all_at_05']})
cvt = pd.DataFrame(rows).set_index('fold') if rows else pd.DataFrame()
if len(cvt): cvt.loc['mean'] = cvt.mean()
cvt.round(4)""")

code("""# all held-out predictions together (each image predicted by the model that did not see it)
parts = [np.load(R / 'preds' / f'wam256_ft_f{k}.npz') for k in range(5) if (R / 'preds' / f'wam256_ft_f{k}.npz').exists()]
pre = np.load(R / 'preds' / 'wam256_pre_train.npz') if (R / 'preds' / 'wam256_pre_train.npz').exists() else None
P = {}
for z in parts:
    for n, p, a in zip(z['names'], z['prob'], z['agree']): P[str(n)] = (p.astype(float), a.astype(float))
t = df.loc[list(P)].copy() if P else df.iloc[:0].copy()
t['miou'] = [miou_image(P[n][0] > 0.5, gts[n]) for n in t.index]
if len(t): display(t.pivot_table(index='method', columns='regime', values='miou', aggfunc='mean').assign(all=t.groupby('method').miou.mean()).round(4))""")

md("## 4. Image · ground truth · probability · agreement · prediction (held-out)")
code("""def panel(names, title):
    fig, axs = plt.subplots(len(names), 5, figsize=(13, 2.7 * len(names)))
    for r, n in enumerate(names):
        p, a = P[n]
        axs[r, 0].imshow(ds[idx[n]]['image']); axs[r, 0].set_ylabel(f"{df.loc[n].method}\\n{df.loc[n].regime}", fontsize=8)
        axs[r, 1].imshow(gts[n], cmap='gray', vmin=0, vmax=1)
        axs[r, 2].imshow(p, cmap='magma', vmin=0, vmax=1); axs[r, 3].imshow(a, cmap='RdBu_r', vmin=-1, vmax=1)
        axs[r, 4].imshow(p > 0.5, cmap='gray', vmin=0, vmax=1); axs[r, 4].set_title(f"mIoU {miou_image(p > 0.5, gts[n]):.2f}", fontsize=9)
        for x in axs[r]: x.set_xticks([]); x.set_yticks([])
    for x, s in zip(axs[0], ['image', 'GT', 'P(watermarked)', 'agreement', 'pred (p>0.5)']): x.set_title(s if s != 'pred (p>0.5)' else x.get_title(), fontsize=9)
    fig.suptitle(title); plt.tight_layout(); plt.show()

rng = np.random.default_rng(0)
for m in data.METHODS:
    tm = t[t.method == m]
    if not len(tm): continue
    panel(list(rng.choice(tm.index, 6, replace=False)), f'{m}: random held-out images')
    o = tm.sort_values('miou'); panel(list(o.index[:3]) + list(o.index[-3:]), f'{m}: 3 worst and 3 best')""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("02_network.ipynb"); nbf.write(nb, out); print("wrote", out)
