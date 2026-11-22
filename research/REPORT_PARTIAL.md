# Partial images — RS data for rectangles and objects (Sun 22 Nov 2026, 03:45; updated as runs finish)

Companion of `research/REPORT_FULL.md`. Same method (real-domain synthetic data, "RS"), other half of the test set.
**[measured]** = clean single-model runs on the real partial train images, same 5 folds. Nothing was submitted.

## 1. Verdict

**RS data makes the outpainting rectangles visible to the segmenter for all three methods, with no real partial label
and no zoom.** Until now the maps did not see PixelSeal and StegaStamp rectangles at all.

| within-image AUC on real rectangles **[measured]** | old zero-shot | current ensemble | zoom map | RS-partial, zero-shot |
|---|---|---|---|---|
| TrustMark (44 rectangles) | 0.682 | 0.765 | 0.929 | **0.980** |
| PixelSeal (43) | 0.439 | 0.592 | 0.589 | **0.835** |
| StegaStamp (44) | 0.708 | 0.656 | 0.850 | **0.874** (0.940 as max with its own zoom map) |

In mIoU the gain is real but smaller, because a blurry blob on a 60 × 90 px rectangle still scores poorly:

| threshold-only CV on the 85 partial images **[measured]** | current ensemble | RS-partial zero-shot | RS-partial fine-tuned | mean(ensemble, fine-tuned, zero-shot) |
|---|---|---|---|---|
| TrustMark: all / rectangles / objects | 0.613 / 0.553 / 0.678 | 0.678 / 0.696 / 0.658 | 0.682 / 0.702 / 0.661 | **0.695** / 0.698 / 0.691 |
| PixelSeal | 0.565 / 0.485 / 0.646 | 0.623 / 0.600 / 0.646 | 0.620 / 0.589 / 0.651 | **0.637** / 0.599 / 0.675 |
| StegaStamp (3000 images; 8000 running) | 0.597 / 0.492 / 0.709 | 0.614 / 0.572 / 0.659 | 0.610 / 0.558 / 0.665 | **0.628** / 0.557 / 0.705 |

What this is worth in your pipeline depends on the method:
- **TrustMark: nothing new.** Your boxes already fix its rectangles: with boxes the combined tag gives 0.7236 against
  0.7243 without the RS-partial maps.
- **PixelSeal and StegaStamp: this is where it pays**, because their boxes mostly fail (PixelSeal decodes 33 % of its
  rectangles; StegaStamp's search does not find them: `zrect` 0.639 → 0.642). Numbers in section 3.
- Objects: RS-partial alone is slightly below the current ensemble; averaged with it, it is level or slightly above.
  SAM still does the shape there.

## 2. How the partial data is made (`tests/realsynth_partial.py`)

- **Base** = a real partial image of the *other two* methods (train, plus test images the gate calls partial).
  It is about 96 % clean; its small watermark belongs to another method, so "clean" is the right label for this
  method's detector. Bases of the same method are not used: their real watermark would be labelled clean.
- **Rectangle** (half of the samples): geometry drawn like the real ones (area 0.3–35 % log-uniform, aspect within the
  method's limit, any position). The base's own crop is upscaled to 512 px, watermarked with the kit embedder, and
  **only the residual is shrunk back and added**. The watermark therefore has the shrunk scale of a real outpainting
  and the image content is untouched — real rectangle borders show no seam (`tests/test_seam.py`).
- **Object** (other half): a real object-shaped partial mask of another method. Half: another base watermarked whole
  and pasted through the mask (a visible foreign object, like a real transplant). Half: the base itself watermarked
  inside the mask (no visible object, so the watermark is the only cue).
- Processing: none or one FLUX round trip of the composite, mixed. Never "pre".
- Model and training: `e2e.Net`, 512 input, default head, 3000 images, 8000 steps; then fine-tune on the real partial
  images of the training folds.
- Option `RS_REALWM=1` (written, not yet run): half of the samples carry the *real* watermark — a real full image of
  the method shrunk into the rectangle, or a region of it pasted through the object mask.

## 3. Ready-made tags (pipeline CV = gate + `postproc.py`, no boxes, no SAM)

| drop-in tag in `research/out/` | built from | all | full | partial |
|---|---|---|---|---|
| `ensRS2_trustmark_{train,test}.npz` | `ensRS_trustmark` + RS-partial maps on gate-partial images | 0.619 → 0.684 | 0.609 → 0.661 | 0.630 → 0.706 |
| `ensRS2_pixelseal_{train,test}.npz` | `ensRS_pixelseal` + RS-partial maps on gate-partial images | 0.586 → 0.648 | 0.595 → 0.657 | 0.577 → 0.638 |
| `ensRS2_stegastamp_{train,test}.npz` | `ensRS_stegastamp` + RS-partial maps on gate-partial images (3000-image partial model) | 0.639 → 0.676 | 0.673 → 0.726 | 0.604 → 0.625 |

The first number of each pair is the tag you use now (`ens2_trustmark`, `ens3_pixelseal`, `ens2_stegastamp`).
Mean of the three methods, all images: **0.615 → 0.669**; with your boxes (`recteval.py`): **0.653 → 0.689**
(TrustMark 0.698 → 0.724, PixelSeal 0.621 → 0.667, StegaStamp 0.639 → 0.676). The partial gains over the full-only tags are
+0.077 (TrustMark, [+0.046, +0.111]), +0.061 (PixelSeal, [+0.034, +0.089]) and +0.023 (StegaStamp, [+0.004, +0.044]).
Built with `python research/tests/make_ensrs.py <method> ensRS_<method> e2eRPzA_<method>,e2eRPfA_<method> ensRS2_<method> 0.333 partial`.

## 4. What I would do with it

1. Run `pipecv.py` with `ensRS2_pixelseal` and `ensRS2_stegastamp` as the map (after copying the files to
   `results/preds/`). Your zoom route ("zoom-map rectangle when the zoom peak beats the map peak") was built for maps
   that do not see rectangles; with these maps it may no longer be needed for PixelSeal, where zoom never worked.
2. Use the RS-partial map as the seed of the box search for PixelSeal and StegaStamp
   (`seedrect.py --maps e2eRPzA_<method> ...`): a top peak of these maps falls inside the rectangle far more often than
   before. For exact rectangles a decoder-verified box still beats any map (`tests/test_rp_rect.py`: a max-sum
   rectangle on the map gives 0.727 on TrustMark rectangles, your boxes 0.87).
3. StegaStamp responds strongly to data size on full images (0.653 → 0.680 → 0.720 for 1200 → 3000 → 8000 images);
   the 8000-image partial model is running and will replace the 3000-image one if it is better.

## 5. Caveats

- 85 partial images per method (about 43 rectangles, 42 objects); differences under 0.02 are noise.
- The regime is taken as known in section 1 (true partial images); the tags in section 3 use the gate.
- Rectangle geometry is drawn from a parametric distribution fitted by eye to the real one, not from real masks.
- Object masks come from 82–83 real masks of the other two methods, with flips and transposes only.
