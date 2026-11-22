# Localising post-hoc image watermarks (PixelSeal / TrustMark / StegaStamp)

Team BlitzTeam — European AI Championship, Paris, 21–22 Nov 2026, task 1 "Localize Image Watermark"
(organised by SprintML / CISPA). Final public leaderboard score **0.739** (mean per-image mIoU over the two classes),
14th of 42.

## The task

Each 512×512 image carries a PixelSeal (256 bits), TrustMark (100 bits) or StegaStamp (100 bits) watermark with a
known message. Edits then removed or added the watermark in parts:
- **full** images: watermarked everywhere except edited holes (median hole 6 % of the image, often several);
- **partial** images: only an outpainted rectangle (the original image shrunk into a larger canvas) or a transplanted
  object carries the watermark (median 4 % of the image).

Output: a 128×128 binary mask per image. Score per image = (IoU of the watermarked class + IoU of the clean class) / 2.

## What worked (5-fold cross-validation on the 510 train images; CV predicted the public score within ±0.006)

| Step | Idea | Effect |
|---|---|---|
| Gate | Decode the whole image with the official decoder; clean decode → full, else partial; later the mean of our own map | 97 % → 99 % regime accuracy |
| Segmenter | The watermark decoder's own early layers (ConvNeXt / ResNet / StegaStamp convs rebuilt from ONNX) fine-tuned end-to-end as a segmenter, at 512 px input (never shrink: small rectangles lose their watermark) | probe 0.53 → 0.63 |
| Synthetic data | Synthetic training images passed through the FLUX autoencoder, then **real-domain synthetic data (RS)**: the dataset's own partial images as bases, real hole / rectangle / object masks, watermark restored or removed as the data loader describes; scaled from 1k to 64k images per model | +0.07 overall; +0.01 per doubling of data |
| Boxes | Decoder-verified rectangle search for outpainted rectangles (crop + resize + known-message logit score, coarse grid + refinement; StegaStamp seeded by RS-partial maps) | TrustMark +0.08 |
| Zoom | Tiles magnified ×2–4 so shrunk watermarks become visible; best rectangle on the zoom map, routed by a map-peak rule | StegaStamp +0.014 |
| SAM | SAM ViT-B clicked at the map peak for transplanted objects, kept when it agrees with the map | +0.011 – 0.019 |
| Ensembles | Largest RS models only (≥ 32k images) averaged with the earlier ensemble; per-regime post-processing | +0.01 – 0.02 |

Full history of submissions with CV and public scores: [`submissions.md`](submissions.md).
Research reports: [`research/REPORT.md`](research/REPORT.md), [`research/REPORT_FULL.md`](research/REPORT_FULL.md),
[`research/REPORT_PARTIAL.md`](research/REPORT_PARTIAL.md). Session notes: [`HANDOFF.md`](HANDOFF.md).

Tried and rejected (do not repeat): TruFor, autoencoder reconstruction error, pixel-level matched filters with the
re-embedded pattern, decoder per-location agreement maps, DINOv2 features, SD2-inpainted holes, 128×128 output head,
768 input, low-resolution RS models, fp32 vs bf16, Lovász loss, scale augmentation, per-image thresholds.

## Main code

| File | Purpose |
|---|---|
| `data.py`, `metrics.py` | data access (wraps the organisers' starter kit), folds, official metric, submission writer |
| `e2e.py` | decoder-based segmenter (training, CV maps, test maps, zoom maps) |
| `research/tests/realsynth_prod.py`, `realsynth_partial.py` | RS data generation + training for full / partial images |
| `bruterect.py`, `seedrect.py`, `research/tests/seedrect_rs.py` | decoder-verified rectangle search |
| `sam_cache.py` | SAM masks prompted at the map peak |
| `postproc.py`, `pipecv.py` | per-regime post-processing; cross-validation of the whole pipeline |
| `integrate.py` | builds ensembles from all finished models and picks one of a fixed set of rules per method by CV |
| `make_sub3.py` | final pipeline on the test set (gate → box → zoom rectangle → SAM → map) |
| `slurm/` | worker scripts used on the JURECA GPU nodes |

The competition data, the starter kit and the watermark model weights belong to the organisers / their authors and are
not included. Paths assume the starter-kit layout (`../release`, `../wmloc`, `../models`).
