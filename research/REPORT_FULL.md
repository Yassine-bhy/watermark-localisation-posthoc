# Full images — answer to FULL_BRIEF.md (Sun 22 Nov 2026, final version 02:50)

Follow-up of `research/REPORT.md`. Scripts: `research/tests/`. Logs and maps: `research/out/`.
**[measured]** = run tonight on the real train images, same 5 folds, full images only. Everything else is reasoning
and is labelled. Nothing was submitted; no existing file or running job was touched.

**If you read an earlier version of this file: two sets of numbers in it were wrong and are withdrawn (section 7).**
Every number below comes from runs that are not affected.

## 1. Verdict

**None of this closes +0.07 to +0.12 by 09:00, but it covers half of it. One change is measured and ready to use:
+0.054 on TrustMark full images, +0.063 on PixelSeal, +0.054 on StegaStamp. Over the three methods the full regime
goes from 0.626 to 0.683 (+0.057) and all images from 0.615 to 0.643 (+0.028), in the pipeline's own CV without boxes
or SAM. With your boxes (`recteval.py`, all images): TrustMark 0.698 → 0.724, PixelSeal 0.621 → 0.65; StegaStamp
(no boxes yet) 0.639 → 0.664.**

The change is a second kind of synthetic training data, built from the dataset's own images ("RS", real-domain
synthetic), whose maps are averaged with the current ensemble on gate-full images. The files exist (state 03:30):

| drop-in tag in `research/out/` | built from | pipeline CV, all images | full | partial |
|---|---|---|---|---|
| `ensRS_trustmark_{train,test}.npz` | `ens2_trustmark` + RS models X and S | 0.619 → 0.646 | 0.609 → **0.663** (+0.054, [+0.040, +0.068]) | 0.630 → 0.629 |
| `ensRS_pixelseal_{train,test}.npz` | `ens3_pixelseal` + RS models X and A | 0.586 → 0.617 | 0.595 → **0.657** (+0.063, [+0.044, +0.083]) | 0.577 → 0.577 |
| `ensRS_stegastamp_{train,test}.npz` | `ens2_stegastamp` (weight 0.2) + RS model XL (8000 images), fine-tuned map only | 0.639 → 0.664 | 0.673 → **0.727** (+0.054, [+0.037, +0.070]) | 0.604 → 0.602 |

"Pipeline CV" is your protocol: decode gate, then `postproc.py`'s per-regime smoothing, operation and threshold, all
fit on the training folds; no boxes, no SAM. Brackets are paired-bootstrap 95 % intervals over the full images.
Copy the two files of a tag to `results/preds/` and the tag works with `postproc.py`, `recteval.py`, `make_sub2.py`
and `pipecv.py` unchanged. Partial images keep the old map, so nothing changes for them.

Everything on the decision side is exhausted (section 5), and so is my list of signals. For full images the remaining
GPU time should go to more RS models (section 4, idea 2), and no further.

## 2. What RS is and why it helps

The current synthetic data differs from the real full images in ways visible in the data:

| | real full images **[measured]** | current synthetic (`synth_flux`, kind holes) **[measured]** |
|---|---|---|
| base image | generated-looking scenes | COCO photos |
| hole components touching the image border | 59 % (79 % of images have one) | 27 % (42 %) |
| hole components per image (128 grid) | 3.25 | 2.05 |
| median hole area | 6.2 % | 9.2 % |
| hole content | nothing visible: SAM, TruFor and the eye find no object or seam | half of the holes are a pasted foreign crop (`synth.py` `holes()`, "addition") |

RS removes these mismatches:
- **base** = a real *partial* image of the dataset (train, plus test images the gate calls partial): about 96 % of
  its area is clean and came out of the organisers' pipeline. 745 bases.
- the **whole frame is watermarked** with the kit embedder and a random message;
- **hole** = a real hole mask of a full train image (flips, transposes, sometimes two masks joined), **restored to the
  base pixels** — the docstring of `wmloc/data.py`, literally;
- optional processing of the composite (recipes below). Model and training are yours: `e2e.Net`, 512 input.
- For method M the hole masks come only from the full train images of the other two methods (170 masks), so no label
  of M's images enters its synthetic data.

## 3. Results, clean runs only  **[measured]**

"Zero-shot" = the model has seen no real label. Its mIoU uses the best single threshold on the 85 full images.
"Fine-tuned CV" = 5-fold, fine-tuned on the real full images of the training folds, one global threshold fit on them.

| TrustMark | images / steps | zero-shot hole AUC | zero-shot mIoU | fine-tuned CV |
|---|---|---|---|---|
| old synthetic (best of three) → `e2eR512`; ensemble `ens2` = 0.613 | 6000 / 6000 | 0.853 | 0.577 | 0.592 |
| RS none (no processing) | 1000 / 2500 | 0.888 | 0.611 | 0.602 |
| RS post (FLUX round trip of the composite), two runs | 1000 / 1500, 2500 | 0.890, 0.873 | 0.602, 0.600 | 0.593, 0.594 |
| RS du384 (composite resized to 384 px and back) | 1000 / 2500 | 0.883 | 0.612 | 0.598 |
| RS mix of none, post, pre — two data seeds (maps A, B) | 1500 / 6000 | 0.881, 0.878 | 0.614, 0.624 | 0.611, 0.628 |
| RS mix of du384, jpg90, blur07, str60 (map C) | 1500 / 5000 | 0.881 | 0.603 | 0.602 |
| **RS mix of none and post, larger (map S)** | **3000 / 8000** | **0.898** | **0.642** | **0.641** |
| **RS mix of none and post, largest (map X)** | **6000 / 16000** | **0.912** | **0.647** | **0.649** |
| RS pre (round trip first, holes restored after), two runs | 1000 / 2500; 3000 / 6000 | 0.821, 0.817 | 0.539, 0.488 | 0.581, 0.593 |
| RS post with `--deep` | 1000 / 2500 | 0.860 | 0.580 | 0.607 |

| PixelSeal | images / steps | zero-shot hole AUC | zero-shot mIoU | fine-tuned CV |
|---|---|---|---|---|
| old synthetic (best of two) → `e2eR512`; ensemble `ens3` = 0.594 | 6000 / 6000 | 0.819 | 0.545 | 0.556 |
| RS none | 1000 / 2000 | 0.880 | 0.584 | 0.602 |
| RS du384 | 1000 / 2000 | 0.896 | 0.582 | 0.596 |
| RS post, two data seeds (the second is map A) | 2000 / 4000 | 0.906 | 0.608 | 0.630, 0.623 |
| RS mix of none and post (map B) | 2000 / 4000 | – | – | 0.620 |
| **RS mix of none and post, larger (map X)** | **4000 / 8000** | **0.928** | **0.618** | **0.629** |
| RS pre | 1000 / 2000 | 0.608 | 0.263 | 0.483 |

| StegaStamp | images / steps | zero-shot hole AUC | zero-shot mIoU | fine-tuned CV | mean(`ens2`, fine-tuned) |
|---|---|---|---|---|---|
| old synthetic → `e2eR512`; ensemble `ens2` = 0.669 | 6000 / 6000 | 0.872 | 0.548 | 0.658 | |
| RS none / post / pre | 1200 / 3000 | 0.887 / 0.897 / 0.892 | 0.563 / 0.570 / 0.567 | 0.647 / 0.638 / 0.643 | 0.679 / 0.675 / 0.675 |
| RS hires1024, str60, blur07, du384, jpg90 | 1200 / 3000 | 0.87 – 0.89 | 0.550 – 0.573 | 0.637 – 0.647 | 0.675 – 0.677 |
| RS mix of none and post (map A) | 1200 / 3000 | 0.895 | 0.569 | 0.653 | |
| **RS mix of none and post, larger (map X)** | **3000 / 8000** | **0.902** | **0.567** | **0.680** | |
| **RS mix of none and post, largest (map XL)** | **8000 / 20000** | **0.920** | **0.582** | **0.720** | |

What the tables say:
- **RS transfers better than the old synthetic data** without any real label: TrustMark 0.60–0.62 against 0.577,
  PixelSeal 0.58–0.61 against 0.545, StegaStamp 0.55–0.57 against 0.548.
- **A single RS model, fine-tuned, beats the current ensemble for all three methods** once it is large enough:
  PixelSeal 0.62–0.63 against 0.594; TrustMark 0.649 against 0.613; StegaStamp 0.720 against 0.669.
- **Size helps, and most for StegaStamp** (fine-tuned CV, one run per size): StegaStamp 1200 images / 3000 steps →
  0.638–0.653; 3000 / 8000 → 0.680; 8000 / 20000 → 0.720. TrustMark 1000 / 2500 → 0.594–0.602; 1500 / 5000–6000 →
  0.602–0.628; 3000 / 8000 → 0.641; 6000 / 16000 → 0.649. PixelSeal 1000 / 2000 → 0.602; 2000 / 4000 → 0.620–0.630;
  4000 / 8000 → 0.629 (flat).
- For StegaStamp the zero-shot map stays poor at any size (0.57); only its fine-tuned map is worth using.
- **The processing recipe hardly matters** (none, post, du384, mixes: all within run-to-run spread, about ±0.01).
  The gain comes from the bases and the hole shapes. The one exception is **"pre", which collapses** for both
  methods: in "pre" the hole is untouched and the rest is round-tripped, so the network learns to spot the round trip
  and ignores the watermark.
- More context (`--deep`) is worse.
- Fine-tuning on real full images does little for TrustMark and adds about 0.02 for PixelSeal, 0.08 for StegaStamp.

## 4. Ranked ideas

### Idea 1 — Use the RS maps on gate-full images  (measured, passes, files ready)

**Mechanism.** Section 2. In the ensemble the RS models also make different errors from the current ones, because
they learned from different images.

**Implementation.** Done for the three methods: copy `research/out/ensRS_<method>_{train,test}.npz` to
`results/preds/` and use the tag `ensRS_<method>` wherever you use `ens3_pixelseal` / `ens2_trustmark` /
`ens2_stegastamp`.
To rebuild or change the mix: `python research/tests/make_ensrs.py <method> <old_tag> <rs_tag,rs_tag,...> <out_tag>
[old_weight]` (CPU, about a minute; it prints the pipeline CV for old and new). The RS maps are
`research/out/e2eRSz<X>_<method>` (zero-shot) and `e2eRSf<X>_<method>` (fine-tuned), train and test.
If your current best ensemble is no longer `ens2` / `ens3`, pass that tag as `<old_tag>`.

**The RS models are holes-only specialists.** Their maps are meaningless on partial images; `make_ensrs.py` only uses
them where the gate says full.

**Cost.** None beyond what is done. **Expected gain.** As in section 1.

**Kill test** (thresholds written in `test_realsynth.py` before the first run): zero-shot mIoU ≥ best old zero-shot
+ 0.03, or fine-tuned CV ≥ `e2eR512` + 0.02, or mean(ensemble, new) ≥ ensemble + 0.015.
TrustMark: the larger model passes all three (0.642 vs 0.607; 0.641 vs 0.612; pipeline +0.043); the small ones pass
the first only in some runs (0.600–0.624). PixelSeal passes all three (0.608 vs 0.575; 0.630 vs 0.576; pipeline
+0.052). StegaStamp: the small models fail all three (0.573 vs 0.578; 0.653 vs 0.678; pipeline +0.009 vs +0.015);
the larger one passes the second and third (0.680 vs 0.678; pipeline +0.019).

### Idea 2 — Larger RS sets  (measured for the three methods: passes; already in the tags)

**Mechanism.** An RS model is limited by how many base × hole × message combinations it has seen, not by the
recipe. Unlike the old synthetic data, where 24k steps overfitted, RS data matches the real images, so more of it and
longer training should keep paying.

**Numbers [measured], pipeline CV on full after folding each larger model in.**
TrustMark: 0.644 (1500-image models) → 0.652 (3000 images / 8000 steps) → 0.663 (6000 / 16000).
PixelSeal: 0.647 (2000 / 4000) → 0.657 (4000 / 8000).
StegaStamp: 0.682 (1200 / 3000) → 0.693 (3000 / 8000) → 0.727 (8000 / 20000).
Adding small models of the same size does nothing (TrustMark four → six maps: 0.643 → 0.644; PixelSeal three → five
maps: 0.651 → 0.647). The steps shrink (TrustMark +0.008, then +0.011 with two models; fine-tuned single-model CV
0.641 → 0.649), so expect each further doubling to bring about half as much.

**Implementation.** `python research/tests/realsynth_prod.py <method> <recipe> <steps> <N> <tag> <seed>` writes the
four map files of one model. Use `mix:none+post` or `post`; never `pre`. Then `make_ensrs.py`.
The three tags were built with:
`python research/tests/make_ensrs.py trustmark ens2_trustmark e2eRSzX_trustmark,e2eRSfX_trustmark,e2eRSzS_trustmark,e2eRSfS_trustmark ensRS_trustmark`
`python research/tests/make_ensrs.py pixelseal ens3_pixelseal e2eRSzX_pixelseal,e2eRSfX_pixelseal,e2eRSzA_pixelseal,e2eRSfA_pixelseal ensRS_pixelseal`
`python research/tests/make_ensrs.py stegastamp ens2_stegastamp e2eRSfXL_stegastamp ensRS_stegastamp 0.2`
(StegaStamp: fine-tuned map only, old ensemble at weight 0.2; its zero-shot map and the old ensemble both pull it down.)
Next size steps are listed in `research/NIGHT_LOG.md`.

**Cost.** TrustMark 9 min (3000 / 8000) to 25 min (6000 / 16000); PixelSeal 12 to 25 min; StegaStamp 3 to 6 min.

**Expected gain.** Another +0.005 to +0.02 on TrustMark and PixelSeal full images — a guess from one size step.

**Kill test.** After folding a model in, the pipeline CV on full printed by `make_ensrs.py` must rise by at least
0.005 over the tag it replaces (now TrustMark 0.663, PixelSeal 0.657, StegaStamp 0.727). If it does not, keep the
current files and stop scaling that method. Passed so far at every step for the three methods.

### Ideas 3 to 5 — none

I have nothing else with evidence behind it. What I tested and dropped tonight is in section 5; I would rather
give you two ideas that hold than five that do not.

**Order.** Take the three ready tags now and run your full pipeline CV (boxes, zoom, SAM) with them; that needs no
GPU. If they do not reproduce the gain inside your pipeline, stop there and tell me what `pipecv.py` printed.
What I am doing next is in `research/NIGHT_LOG.md`: the same RS idea for partial images
(`tests/realsynth_partial.py`, first test running), one more size step where it still pays, and a diagnostic of what
the RS models still miss (`tests/rs_diag_synth.py`).

## 5. The four angles of the brief

**Signals beyond the early layers.** None found. Tested tonight **[measured]**: no JPEG history (8-px grid ratio
1.02–1.04; block-DCT near-zero count, hole AUC 0.54–0.61); FLUX reconstruction error is higher in the watermarked area
than in the hole in only 60–72 % of images; the image spectrum shows no resize cut-off. Whether the images went
through a FLUX VAE stays open: a first FLUX round trip changes them by a median of 2.15 / 255, a second by about 1.5,
a COCO photo by 3.0–3.6 — in between, so no conclusion.

**Finding all the holes.** The decision side cannot do it **[measured]**:
- a greedy *oracle* picking up to 4 blobs from a threshold ladder gains +0.075 to +0.10, so part of the holes is there;
- a learned blob rescoring (11 features, small MLP, same folds) gains +0.001 (`test_comprescore.py`, pass was +0.02);
- per-image thresholds from expected mIoU under calibrated probabilities gain +0.003 (`test_expiou.py`, pass +0.01);
- 28–46 % of true hole components match no blob at any threshold.
Iterative "find, mask, re-decode" has no basis: the decoder's global score does not respond locally
(`test_counterfactual.py`). The holes have to appear in the map.

**Public datasets.** None needed, none found with these watermarks. The dataset itself supplies the bases and the
hole shapes, which is what RS uses.

**Something simple the 0.77 teams found.** I do not know. Ruled out **[measured]**: hidden linkage (all 1530
method–message pairs are distinct, none shared between train and test), JPEG traces, mask grid alignment (edges on
the 8-px grid at chance, 12.5 %), a position prior (centre 8.2 % vs outer ring 9.8 %), a metric quirk (every full
train image has a hole, so the empty-union bonus never applies). My guess is better-matched training data and a
larger model trained longer; RS is a step of that kind, and it bought 0.03, not 0.10.
Also tested: your holes-only model on the old synthetic data, `e2eFH_trustmark`: 0.597 against 0.592, noise level.

## 6. Caveats

- 85 full images per method. The intervals in section 1 are the honest size of the uncertainty.
- `make_ensrs.py` fits the gate threshold on all train images (one parameter); the old-map weight (one third) was
  not tuned.
- The gate decides which test images serve as bases; a few full images slip in (gate accuracy ≈ 97 %).
- My CV numbers contain no boxes and no SAM; the gain on full images should carry over because those steps act on
  gate-partial images, but only your `pipecv.py` can confirm it.
- The project's inode quota ran out around 01:15 (`Disk quota exceeded` killed the `e2eFHi` and `zrect` saves).
  RS outputs are 4 files per model; nothing writes per-image files.
- GPU use by this session: up to four node GPUs for short periods when they were idle, each run under 25 minutes.

## 7. What was wrong in earlier versions of this file

`e2e.Net` uses the watermark decoder's layers by reference. My scripts that train several variants in one process
(`test_realsynth.py`, `test_realsynth_sim.py`) therefore started every variant after the first from the previous
variant's backbone, which had just been fine-tuned on the real images of four folds. Real labels leaked into the
"zero-shot" and "fine-tuned" numbers of every TrustMark and PixelSeal variant except the first of each run, more with
each position. Fixed at 02:35 (a fresh decoder per variant). Your `e2e.py` is not affected (one model per process),
nor `realsynth_run.py`, `realsynth_prod.py`, nor any StegaStamp number (its backbone is rebuilt from the ONNX file).

Withdrawn because of it:
- TrustMark "pre" at 0.638 / 0.637 / 0.647 (first version) and 0.648 (later): clean runs of "pre" collapse.
- The table posted at 02:30 in which softened watermarks (du384 0.653 / 0.677, jpg90, blur07, str60, hires1024)
  beat the FLUX round trip: a clean du384 run gives 0.612 / 0.598, the same as the others.
- "More steps help" and the three-recipe mix at 0.634 / 0.643 / 0.663.
- PixelSeal post at 1000 images (0.611 / 0.615).

In `research/out/`, logs written before 02:35 by those two scripts are valid for their first variant only:
`test_realsynth_{trustmark,pixelseal}.log`, `test_realsynth_sim_trustmark.log`, `test_realsynth_curve_trustmark.log`,
`test_realsynth_sim2_{trustmark,pixelseal}.log`, `test_realsynth_sim3_trustmark.log`. The map files those runs
wrote (`realsim_*.npz`, `realsynth_*_train.npz`) have been deleted.
