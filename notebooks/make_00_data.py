"""Builds notebooks/00_data.ipynb (run: python notebooks/make_00_data.py; then it is executed with nbconvert)."""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 00 — Look at the data
Train images and their ground-truth masks, per method and per regime
(`full_watermark=False` → *partial*: a watermarked rectangle in a clean image;
`full_watermark=True` → *full*: whole image watermarked, then some regions restored to clean).
Everything here is recomputed from the files, nothing is copied from chat.""")

code("""import sys; sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import ndimage as ndi
import data, metrics
pd.set_option('display.width', 140)
df = data.train_table()
gts = data.train_masks128()
ds = data.load_split('train')
idx = {m['image_name']: i for i, m in enumerate(ds.meta)}
df.head()""")

md("## 1. How many images of each kind")
code("""pd.crosstab(df.method, df.regime, margins=True)""")

md("## 2. Watermarked fraction of each image (128 grid)")
code("""fig, axs = plt.subplots(1, 3, figsize=(15, 3.2), sharey=True)
for ax, m in zip(axs, data.METHODS):
    for r, c in [('partial', 'tab:blue'), ('full', 'tab:orange')]:
        ax.hist(df[(df.method == m) & (df.regime == r)].wm_frac, bins=np.linspace(0, 1, 41), color=c, alpha=.7, label=r)
    ax.set_title(m); ax.set_xlabel('watermarked fraction')
axs[0].legend(); plt.show()
df.groupby(['method', 'regime']).wm_frac.describe().round(3)""")

md("""## 3. Gallery
For each method × regime, 5 random images. Top: image with the **watermarked** area tinted red.
Bottom: the official 128×128 mask (white = watermarked).""")
code("""def gallery(method, regime, n=5, seed=0):
    names = df[(df.method == method) & (df.regime == regime)].sample(n, random_state=seed).name
    fig, axs = plt.subplots(2, n, figsize=(3 * n, 6.2))
    for j, name in enumerate(names):
        it = ds[idx[name]]
        over = it['image'].astype(float) / 255
        over[it['mask'] == 1] = 0.55 * over[it['mask'] == 1] + 0.45 * np.array([1, 0, 0])
        axs[0, j].imshow(over); axs[0, j].set_title(f'{name}\\nwm={gts[name].mean():.2f}', fontsize=9)
        axs[1, j].imshow(gts[name], cmap='gray', vmin=0, vmax=1)
        for a in axs[:, j]: a.axis('off')
    fig.suptitle(f'{method} — {regime}', fontsize=14); plt.tight_layout(); plt.show()

for m in data.METHODS:
    for r in ('partial', 'full'):
        gallery(m, r)""")

md("""## 4. Shape of the minority region
Minority region = watermarked area for *partial*, clean area for *full*.
`bbox_fill` = share of its bounding box that it covers (1.0 = a perfect rectangle);
`n_comp` = number of connected pieces; `aspect` = long/short side of the bounding box.""")
code("""def shape_stats(name, full):
    g = gts[name]; c = (1 - g) if full else g
    lab, n = ndi.label(c)
    ys, xs = np.nonzero(c)
    h, w = ys.ptp() + 1, xs.ptp() + 1
    return dict(minority_area=c.mean(), bbox_fill=c[ys.min():ys.max()+1, xs.min():xs.max()+1].mean(),
                n_comp=n, aspect=max(h, w) / min(h, w))
st = pd.concat([df, pd.DataFrame([shape_stats(r.name, r.full_watermark) for r in df.itertuples()])], axis=1)
st.groupby(['method', 'regime'])[['minority_area', 'bbox_fill', 'n_comp', 'aspect']].median().round(3)""")

md("""### 4a. What kind of shape is the watermarked region in *partial* images?
Rough rule: `bbox_fill ≥ 0.9` → axis-aligned rectangle; 0.6–0.9 in one piece → rotated or clipped rectangle;
otherwise → object-shaped (e.g. a watermarked object transplanted into a clean image).""")
code("""p = st[st.regime == 'partial'].copy()
p['kind'] = np.select([p.bbox_fill >= 0.9, (p.bbox_fill >= 0.6) & (p.n_comp == 1)],
                      ['rectangle', 'rotated/clipped rect'], 'object-shaped')
pd.crosstab(p.method, p.kind, margins=True)""")

md("""### 4b. Partial images that are *not* clean rectangles (lowest `bbox_fill`)""")
code("""def show(names, title):
    fig, axs = plt.subplots(2, len(names), figsize=(3 * len(names), 6.2))
    for j, name in enumerate(names):
        it = ds[idx[name]]; r = st[st.name == name].iloc[0]
        axs[0, j].imshow(it['image']); axs[0, j].set_title(f'{r.method}\\nfill={r.bbox_fill:.2f} comp={r.n_comp}', fontsize=9)
        axs[1, j].imshow(gts[name], cmap='gray', vmin=0, vmax=1)
        for a in axs[:, j]: a.axis('off')
    fig.suptitle(title); plt.tight_layout(); plt.show()
show(st[st.regime == 'partial'].nsmallest(6, 'bbox_fill').name.tolist(), 'partial, least rectangular')""")

md("""### 4c. TrustMark partial images with aspect > 2
The task says TrustMark rectangles have aspect ≤ 2:1 — check whether the masks agree.""")
code("""odd = st[(st.method == 'trustmark') & (st.regime == 'partial') & (st.aspect > 2.2)]
print(len(odd), 'of', ((st.method == 'trustmark') & (st.regime == 'partial')).sum(), 'TrustMark partial images have aspect > 2.2')
if len(odd): show(odd.nlargest(6, 'aspect').name.tolist(), 'TrustMark partial, largest aspect')""")

md("## 5. Constant baselines (official mIoU on train, from `baselines.py`)")
code("""res = pd.read_csv(data.RESULTS / 'step0_baselines.csv')
for b, g in res.groupby('baseline', sort=False):
    print('==', b); display(metrics.breakdown(g))""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("00_data.ipynb")
nbf.write(nb, out); print("wrote", out)
