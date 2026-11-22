# Brief for the next research round: FULL images only (Sun 22 Nov 2026, ~01:00)

To the author of `research/REPORT.md`. Your two partial-image routes are being integrated (status in section 5).
This round is about the one thing your report could not solve: **the full regime** (watermark rectangle = whole image,
holes = edited regions). It is half of the score, and the top of the leaderboard depends on it.

## 1. Why it matters (arithmetic)

- Leaderboard (public, ~01:00): 1st 0.77, 5th 0.75, 10th 0.73; we are 18th at 0.6456 (CV 0.653).
- Score = mean over images of (IoU_watermarked + IoU_clean) / 2. Full and partial are 50 / 50 of the images.
- Our CV per regime: full **0.595 (PS) / 0.609 (TM) / 0.673 (SS)**, mean ≈ 0.626; partial ≈ 0.62-0.78 and rising.
- A team at 0.77 with partial at ~0.85 needs full ≈ 0.69; with partial 0.80, full ≈ 0.74. **We need full +0.07 to +0.12.**

## 2. What a full image is (measured on the 255 full train images, 85 per method)

- Every full train image has at least one hole (0 / 255 without). 54 % have 2 or more hole components
  (PS 49/85, SS 43/85, TM 49/85). Median hole area 5.6-6.4 % of the image; a quarter under 2.3 %; 10 % under 0.7 %.
- `wmloc/data.py` docstring: full images had "a part restored to the clean original" (your report, section 4).
  So holes may be clean original pixels, not generated content — but whether the whole image then went through a
  VAE / generator is unknown. Evidence for a FLUX-VAE-like round trip: fresh watermarks only look like the real ones
  (pixel-exact pattern weak, decoding intact) after a FLUX VAE round trip (`vae_test.py`); your `test_vae_level.py`:
  second-round-trip change 1.9-2.3/255 vs 3.0-3.6 for a first trip on COCO.
- Images are 512 x 512 PNG, look AI-generated, no metadata, not from COCO (thumbnail retrieval: median best corr 0.67).

## 3. Our best full-image model and where it loses (all 5-fold held-out CV on the real train)

Model: `e2e.py --res 512` = the decoder's own early layers (PS ConvNeXt stem + stages 0-1; TM ResNet stem + layer1-2;
SS dec_conv_1..7 rebuilt in torch) fine-tuned end-to-end as a segmenter: pretrain on 6000 synthetic images per method
(COCO + kit embedders + COCO-object holes / transplants / outpaint rectangles, then FLUX VAE round trip), fine-tune
per fold on the real train. Head: 1x1 + three 3x3 convs at 64 x 64, upsampled to 128. Loss BCE + Dice.
Ensembles: TM / SS = mean of 256-input and 512-input models; PS = mean of e2e 512, e2e 256 and a frozen-feature probe.
Post-processing per regime (Gaussian smoothing, fill / largest component, threshold), chosen in CV.

- Within-image AUC on full images: 0.89-0.93. The map ranks cells well; the masks are still poor.
- Groups (`diag_full.py`): hole missed entirely 58 images (23 %, score ~0.48); hole found but IoU_clean < 0.5:
  117 (46 %, ~0.59); fine 80 (31 %, ~0.78).
- Bad-shape breakdown (`shape_breakdown.py`, 117 images): "missed one of several holes" is the largest loss (19.3 image
  points), then misplaced / wrong shape (12.5), too big / blurry (7.7), merged (4.5), too small (4.2).
- By hole size: small < 3 %: 0.51-0.55 (≈ an empty prediction); medium 3-10 %: 0.58-0.71; large > 10 %: 0.66-0.74.
- Oracles: best threshold per image +0.038 (to 0.664); perfect location with our shapes +0.020; true shape placed at
  our location −0.022. So the map itself must improve; thresholds and post-processing are saturated.

## 4. Everything already tried for full images (do not repeat; numbers are CV or within-image AUC on full images)

| idea | result |
|---|---|
| Post-processing rules: never-empty hole, top-k components by summed confidence, min component size (`step1_full.py`) | +0.005 PS only, otherwise ±0.002 |
| Map-mean gate instead of decode gate | gate accuracy up (TM 0.988, SS 0.994) but end-to-end no gain |
| Per-pixel matched filter: local corr. of high-pass(image) with t = Emb(y,m) − mean Emb(y,m') (`pattern_test.py`) | AUC 0.51-0.52 |
| Your `test_counterfactual.py` (complementary message re-embedding + RISE) | AUC 0.56-0.67 |
| Decoder per-location logits (known-message agreement), shift-and-stitch to 64 x 64 (`stitch.py`) | AUC 0.61-0.64 |
| Extra probe channels: message-signed per-position logits, agreement maps | CV down |
| TruFor off the shelf (`trufor_maps.py`, all 255 full images) | AUC 0.44-0.56 |
| FLUX / SD / SDXL VAE reconstruction error (AEROBLADE-style) | AUC 0.50-0.61 |
| TrustMark's own watermark remover: energy of x − remover(x) | AUC 0.54 (full) |
| DINOv2 features (content) | CV 0.45 |
| SAM point-prompted at the hole (your test) | below baseline (−0.03) |
| Input 768 / no left-right flip / fp32 instead of bf16 / Lovász loss | 0.595-0.602 vs 0.602 (no gain) |
| 128 x 128 output head (`--out 128`) | worse: TM 0.597 vs 0.609, SS 0.610 vs 0.627 |
| Frozen decoder BatchNorm (TM) | +0.006 (noise level) |
| 4x longer synthetic pretraining (24k steps) | worse (overfits synthetic) |
| WAM-style network trained from scratch / fine-tuned (earlier) | CV 0.45 |

Not yet measured: realistic synthetic holes made by SD2 inpainting + FLUX round trip (`inp_*.py`, 3 shards x 400
images per method exist in `results/synth_inp/`); a holes-only specialised model (your idea 3.3); seed ensembles
(running); vertical-flip TTA (maps `<tag>V` being written).

## 5. Status of your partial-image routes (for context)

- Zoom: integrated in `e2e.py` (maps `<tag>Z`, mean of zooms 2-4). Full 5-fold CV confirms: rectangle AUC TM 0.74 →
  0.93, SS 0.67 → 0.84 (object 0.62-0.78, full 0.62-0.75 at zoom: routing needed).
- Box search seeded from the current maps failed as you predicted (SS 0.6351 vs 0.6394). Running now: seeds from zoom
  maps, 4 px steps, scales 1.0-1.2 (`seedrect.py --tag zrect`).
- SAM for objects: SAM masks cached (`sam_cache.py`), honest CV through the real pipeline in `pipecv.py` (gate → box →
  SAM if agree ≥ τ → map, thresholds fit on training folds) — result pending.

## 6. Constraints

- Time: node job ends 11:17 Sunday, reservation 14:00. Results needed by ~09:00 to leave time to integrate and submit.
- Compute: 2 of 4 A100s on one node (teammates share it), one Slurm job per team (no new jobs).
  CPU login node available. Rate limit: 1 submission / 30 min.
- Data: 510 train images (255 full), 1020 test. Anything external is allowed (models, weights, papers).
- Packages: shared `.venv` (torch 2.5.1, diffusers 0.31, timm 1.0, onnx2torch, segment-anything); a separate
  `benelhajyahia1/.venv_gen` with transformers 4.49 for generative models. Never break the shared `.venv`.
- Every claim must be checked on the real train images with the same 5 folds (`data.train_table().fold`), full images
  only, and reported against the baseline 0.595 / 0.609 / 0.673.

## 7. The question

**How do we raise the full-regime mIoU from ≈ 0.63 to ≥ 0.72 by ~09:00 with the compute above?**
Please search widely (watermark localisation, tamper localisation with robust watermarks, inpainting / splicing
localisation in fully regenerated images, segmentation of tiny regions, test-time tricks, anything else), and for each
candidate give: the mechanism, why it should work *given section 2-4* (in particular: decoders still decode perfectly,
pixel-exact patterns are gone, image-content forensics are at chance), expected gain, cost in hours on 2 A100s, and a
kill test runnable in under 30 minutes with a pass threshold written in advance, as in your last report.
Specific open questions:
1. What signal can distinguish a restored clean-original hole from watermarked surroundings when both went through the
   same VAE, other than what the decoders' early layers already see? Is there a better way to extract it (e.g. larger
   receptive fields, multi-scale fusion, iterative decoding with masks, optimisation-based attribution that
   re-runs the decoder with candidate masks, gradient w.r.t. a learnable occlusion mask)?
2. Since many images have 2-3 holes and "missed one of several holes" is the biggest loss: is there a principled way to
   find all holes (e.g. iterative: find a hole, inpaint / mask it, re-decode, repeat)?
3. Does any public model or dataset match this task closely enough to pretrain on (local watermark tamper benchmarks,
   e.g. markbench, EditGuard / OmniGuard data)?
4. Is a teams-at-0.77 result explainable by something simple we overlooked (data leakage, test-set structure, metric
   quirk, image statistics), and how would we check it quickly?
