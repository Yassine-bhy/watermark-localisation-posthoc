# Night log — RS (real-domain synthetic) work, Sun 22 Nov 2026

Running notes of the session that wrote `REPORT_FULL.md`. Newest entries at the bottom. Times are Paris time.
Rules followed: nothing submitted; nothing outside `research/` written; at most 2 node GPUs from 03:10 on
(more only while the node was idle); GPU status checked before every launch.

## What is ready to use (state 03:45)

Drop-in map tags in `research/out/` (copy `<tag>_{train,test}.npz` to `results/preds/`). `ensRS_*` changes full
images only; `ensRS2_*` = `ensRS_*` plus RS-partial maps on gate-partial images. Pipeline CV = gate + `postproc.py`.

| tag | content | pipeline CV, all images (old → new) | full | partial | with your boxes (`recteval.py`), all |
|---|---|---|---|---|---|
| `ensRS_trustmark` | `ens2_trustmark` + RS full models X (6000 images) and S (3000) | 0.619 → 0.646 | 0.609 → 0.663 | unchanged | 0.698 → 0.724 |
| `ensRS_pixelseal` | `ens3_pixelseal` + RS full models X (4000) and A (2000) | 0.586 → 0.617 | 0.595 → 0.657 | unchanged | 0.621 → 0.653 |
| `ensRS_stegastamp` | 0.2 × `ens2_stegastamp` + 0.8 × RS full model XL (8000), fine-tuned map | 0.639 → 0.664 | 0.673 → 0.727 | unchanged | 0.639 → 0.664 |
| `ensRS2_trustmark` | + RS-partial model A | 0.619 → 0.684 | 0.661 | 0.630 → 0.706 | 0.724 (boxes already cover it) |
| `ensRS2_pixelseal` | + RS-partial model A | 0.586 → 0.648 | 0.657 | 0.577 → 0.638 | 0.621 → 0.667 |
| `ensRS2_stegastamp` | + RS-partial model A (3000 images) | 0.639 → 0.676 | 0.726 | 0.604 → 0.625 | 0.639 → 0.676 |

Mean of the three `ensRS2` tags with your boxes: **0.653 → 0.689** (TrustMark 0.724, PixelSeal 0.667, StegaStamp 0.676).
Your `pipecv.py` (boxes + zoom + SAM) was at 0.670 with the old maps; I have not run it with the new ones.

Scripts: `tests/realsynth_prod.py` (one RS full model → 4 map files), `tests/realsynth_partial.py` (same for partial
images), `tests/make_ensrs.py` (adds RS maps to an ensemble on gate-full or gate-partial images, prints the pipeline CV).

## Log

- 01:05–01:20 CPU checks for the full regime: zero-shot gap of the old synthetic (AUC 0.77–0.87), real hole shapes
  (59 % of components touch the border, 3.25 per image), component oracle +0.08–0.10, learned rescoring +0.001,
  expected-mIoU thresholds +0.003, no JPEG trace, messages all distinct.
- 01:20–02:30 first RS tests. **Bug found at 02:35**: multi-variant scripts reused the previous variant's backbone
  (`e2e.Net` shares the decoder's layers) → non-first variants inflated. All conclusions rebuilt from clean runs.
- 02:35–03:00 clean single-model runs: recipe does not matter (none / post / du384 / mixes within ±0.01), "pre"
  collapses, `--deep` worse. Size matters: TrustMark fine-tuned CV 0.60 (1000 images) → 0.641 (3000) → 0.649 (6000);
  StegaStamp 0.653 (1200) → 0.680 (3000); PixelSeal 0.602 (1000) → 0.623–0.630 (2000) → 0.629 (4000).
- 03:00 three tags written and checked (same names and order as the old tags, partial images unchanged).
- 03:10 user's plan for the rest of the night received. My reading: scale RS (supported by the size trend), extend RS
  to partial images (new), skip the decoder-verifier and message conditioning (measured negative), bigger model last.
- 03:12 `realsynth_partial.py trustmark` started (first RS-partial kill test); StegaStamp 8000 images / 20000 steps
  started.
- 03:15 larger full models folded in. Tags now: TrustMark full 0.609 → 0.663 (models X + S), PixelSeal 0.595 → 0.657
  (X + A), StegaStamp 0.673 → 0.693. With your boxes (`recteval.py`): TrustMark 0.698 → 0.724, PixelSeal 0.621 → 0.65,
  StegaStamp 0.639 → 0.649.
- 03:20 **RS-partial passes for TrustMark** (`research/out/realsynth_partial_trustmark_A.log`), 3000 images, 8000 steps:
  on the 85 real partial images, with no real partial label, rectangle AUC 0.980 (old zero-shot 0.682, `ens2` 0.765,
  zoom map 0.929), object AUC 0.935 (`ens2` 0.951); threshold-only CV on partial 0.678 against 0.613 for `ens2`
  (rectangles 0.696 against 0.553). Fine-tuned on real partial images: 0.682; mean(`ens2`, fine-tuned, zero-shot) 0.695.
  The generator: the base's own crop, upscaled, watermarked, and only the residual shrunk back into the rectangle
  (no seam); objects = real object masks, half pasted foreign watermarked content, half watermark only.
- 03:22 tag `ensRS2_trustmark` = `ensRS_trustmark` + RS-partial maps on gate-partial images: pipeline CV all
  0.646 → 0.684, partial 0.629 → 0.706 (+0.077, [+0.046, +0.111]). **With your boxes it changes nothing for TrustMark**
  (0.7236 against 0.7243): the boxes already fix TrustMark's rectangles. The point of RS-partial is PixelSeal and
  StegaStamp, whose rectangles the current maps do not see (PixelSeal rectangle AUC 0.59) and whose boxes mostly fail.
- 03:24 `test_rp_rect.py trustmark`: an exact max-sum rectangle on the RS-partial map gives 0.727 on rectangles against
  0.705 for a threshold; routing rectangle / blob is worth +0.005 (not significant). For exact rectangles keep the
  decoder-verified boxes and use the RS-partial map as their seed.
- 03:21 `realsynth_partial.py pixelseal` started. StegaStamp follows when its GPU frees.
- 03:27 **StegaStamp full, 8000 images / 20000 steps: fine-tuned CV 0.720** (3000 images: 0.680; `ens2`: 0.669).
  Tag rebuilt: `ensRS_stegastamp` = 0.2 × `ens2` + 0.8 × that model's fine-tuned map → pipeline CV full 0.673 → 0.727.
  StegaStamp is the method that responds most to size, so its next doubling is queued after the partial runs.
- 03:30 with your boxes (`recteval.py`, all images): TrustMark 0.698 → 0.724, PixelSeal 0.621 → 0.653,
  StegaStamp 0.639 → 0.664; mean 0.653 → 0.680. (Your `pipecv.py` adds zoom and SAM on top; not run by me.)
- 03:30 inode quota at 1200 free: deleted 47 of my own obsolete map files in `research/out/`.
- 03:32 **RS-partial, StegaStamp, 3000 images**: rectangle AUC 0.656 (`ens2`) → 0.874, and 0.940 as max(plain, zoom);
  threshold CV on partial 0.597 → 0.614 zero-shot, 0.628 with `ens2`; objects slightly below `ens2` (0.66 against 0.71).
  Passes the AUC test, small in mIoU. Rerunning at 8000 images (`realsynth_partial_stegastamp_XL.log`).
  `test_rp_rect.py stegastamp`: max-sum rectangle on the map gives 0.607 on rectangles (threshold 0.572, `ens2` 0.492).
- Team's own StegaStamp box search seeded by zoom maps (`zrect`): 0.6394 → 0.6419, so StegaStamp rectangles still
  have no working box; the RS-partial map is currently the best thing available for them.
- 03:40 **RS-partial, PixelSeal, 3000 images**: rectangle AUC 0.592 (`ens3`) → 0.835 zero-shot; threshold CV on partial
  0.565 → 0.623 zero-shot, 0.620 fine-tuned, 0.637 as mean(`ens3`, fine-tuned, zero-shot). Zoom does nothing for
  PixelSeal (0.589). `test_rp_rect.py pixelseal`: always fitting a max-sum rectangle gives 0.649 on partial against
  0.628 for a threshold (rectangles 0.628 against 0.599; objects no worse).
- 03:42 **combined tags `ensRS2_<method>`** (RS full maps on gate-full images, RS partial maps on gate-partial images),
  pipeline CV without boxes or SAM, all images: TrustMark 0.619 → 0.684, PixelSeal 0.586 → 0.648,
  StegaStamp 0.639 → 0.676; **mean 0.615 → 0.669**. Details in `REPORT_PARTIAL.md`.
- 03:44 written, not yet run: `tests/seedrect_rs.py` (decoder-verified box search seeded by the RS-partial maps, for
  StegaStamp and PixelSeal rectangles) and the `RS_REALWM=1` option of `realsynth_partial.py`.
- Running: `rs_diag_synth.py` (is the remaining loss on full images data or model?), StegaStamp partial at 8000 images.
- 05:30 (continued by benelhajyahia1's session after this log stopped at 03:44) **submitted** step6: pipeline (boxes +
  zoom + SAM) on `ensRS2_trustmark`, `ensRS2_pixelseal`, `ensRSP_stegastamp` = ensRS2 + RS-partial XL: CV 0.707
  (PS 0.679 / TM 0.732 / SS 0.709) → **public 0.7012**.
  `rs_diag_synth` read: TM small synthetic holes 0.596 < 0.65 → decision rule says test RS_OUTGRID=128 (queued);
  SS small synthetic 0.671 but real 0.494 at threshold 0.12 → fidelity / calibration gap.
  `seedrect_rs.py stegastamp` (train): boxes IoU > 0.8 on 20 % of true rectangles (old 5 %), CV on e2eRPzXL map
  0.5988 → 0.6133 (passes the +0.01 rule); test split running, then PixelSeal.
  Running on the node: TM + PS RS full XL, TM + PS RS partial XL, PS RS full XL2 + SS partial XL2, rsbox.
  Queued: grid-128 test (TM), RS_REALWM partial (PS), SS full 16000 images / 30000 steps.
