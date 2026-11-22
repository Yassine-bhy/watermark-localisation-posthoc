# Watermark localisation — research report (Sun 22 Nov 2026, ~00:40)

State of the repo: `research/00_state.md`. Literature notes: `research/research_notes/Post hoc watermark localisation routes/`.
Kill tests: `research/tests/`, logs in `research/out/` and `output/15902410_*_rs_{zoom,sam}.log`.
Everything marked **[measured]** was run on our 510 train images tonight. Everything else is literature or reasoning and is labelled.
Nothing was submitted, no existing file was changed, no running job was touched.

## 1. Verdict

**0.80 is not plausible from the current state in the time left; 0.70–0.72 is, through two routes that passed their kill tests.**
The first is zoomed inference for outpainting rectangles: the rectangle holds a watermark that was shrunk with the original image, and our segmenter only sees it when the image is magnified about 3× (rectangle AUC 0.68 → 0.91 for TrustMark, 0.61 → 0.81 for StegaStamp).
The second is SAM prompted at the peak of our map for transplanted objects (+0.09 to +0.12 mIoU on those images).
Both act on the partial regime only. The full regime (half of the images, 0.60–0.67) had no passing test: with full unchanged at ≈ 0.62, even a partial regime at 0.90 gives 0.76. A team at 0.78 must have the full regime near 0.70 or better, and we have not found how.

## 2. Ranked ideas

Cells: PS / TM / SS × full / rect (outpainting) / object (transplant). Each of rect and object is 1/12 of the data per method; each full cell is 1/6.

| # | idea | cell it fixes | evidence | expected gain (reasoning) | hours | weights, offline? | main risk | kill test |
|---|---|---|---|---|---|---|---|---|
| 1 | Zoomed tile inference (and scale augmentation) of the e2e segmenter for rectangles | rect, all methods | **[measured]** `test_zoom.py`, held-out folds: TM rect AUC 0.68 → 0.91 (zoom 3), 0.94 (mean of zooms), top-5 peak hit 0.44 → 0.89; SS 0.61 → 0.81, hit 0.13 → 0.53. markbench 2026: PixelSeal evidence 0.36 for 0–3 % regions vs 27.4 for 40–85 % | Rectangles are at 0.49 (PS, SS). Maps that see them give seeds for the box search and a direct mask. If SS reaches its 64 % decodable share at TM box quality: +0.02 overall; PS limited by 33 % decodable | 2–3 | none needed | Zoom destroys object and full maps (AUC 0.94 → 0.59), so it must be applied to rectangles only; n = 15–18 per method | **pass** |
| 2 | SAM (ViT-B) point-prompted at the map peak, used when it agrees with the map | object, all methods | **[measured]** `test_samprompt.py`: object mIoU PS 0.643 → 0.759, TM 0.682 → 0.799, SS 0.708 → 0.804 (`agree` rule 0.730 / 0.793 / 0.804); oracle of 3 masks 0.82–0.84. SAFIRE (AAAI 2025): point prompting F1 62.6 vs 28.0 | +0.10 on 25 % of images ≈ +0.025 overall | 1–2 | `external/sam/sam_vit_b_01ec64.pth`, already local | In-sample threshold baseline; hurts if applied to full (−0.03 to −0.09) or rect (−0.01 to −0.04), so needs the gate and the box decision first | **pass** (objects) / **fail** (full, rect) |
| 3 | Dense box search seeded by zoom-map peaks, 4 px steps, scales ≥ 1 | rect SS, PS | **[measured]** `diag2a.py`: SS true box beats found box in 66 %; score at 4 px shift = 45 %, 8 px = 3 %; ×1.05 keeps 99 %, ×0.9 keeps 28 %. `test_seedhit.py`: current maps put a top-5 peak inside the rectangle in only 23–37 % | Included in row 1's estimate; without new seeds `seedrect.py` cannot work | 1–2 | none | `seedrect.py` stage 1 uses 8 px steps, which jump over the SS peak | seeds from current maps: **fail** |
| 4 | Regime-specialised full model: train on hole-type synthetic only, output 128, holes = clean original restored before any VAE pass | full, all methods | `wmloc/data.py` docstring ("part of it restored to the clean original"); **[measured]** full AUC is 0.90–0.93 yet mIoU 0.60–0.67; best per-image threshold only +0.03–0.05. MaskMark (2025): a pixel-space U2-Net localiser reaches IoU 0.63–0.83 by region size | Unknown. This is the only lever on half the data | 3–5 | none | No test run; `e2e --out 128` is already in the chain and may show the size of the effect | not run |
| 5 | Never predict an empty minority; lower thresholds | full with tiny holes | Arithmetic: empty scores ≈ 0.47–0.495, any overlapping blob > 0.5. **[measured]** 58 of 255 full images miss the hole | Small (≤ +0.005); `step1_full.py` already tests it | 0.5 | none | — | teammate's script, result not logged |
| 6 | Fine-tune a forensic localiser (TruFor / DinoLizer) on our own synthetic | full | TGIF2 (2026): fine-tuned TruFor F1 0.76–0.94 on FLUX fully-regenerated; SD2-only training transfers badly to FLUX (0.08–0.40) | Low: if holes are clean originals there is no generated content to find | 4+ | DinoLizer weights on a Lille Nextcloud link, untested | Wrong premise | not run |
| 7 | Off-the-shelf TruFor maps | full | **[measured]** `test_trufor.py`: hole AUC 0.44–0.50; fusing lowers mIoU. TGIF (2024): TruFor F1 0.83 spliced → 0.19 regenerated | none | — | local | — | **fail** |
| 8 | Straight-seam detector for rectangle borders (image content only) | rect | **[measured]** `test_seam.py`: border z > 3 at 5–17 % of borders | none | — | — | — | **fail** |
| 9 | Exact max-sum rectangle on current maps | rect | **[measured]** `test_maxrect.py`: +0.00 to +0.01 | none on current maps; worth re-running on zoom maps | 0.5 | — | — | **fail** (current maps) |
| 10 | Complementary-message re-embedding + RISE regression | full | **[measured]** `test_counterfactual.py`: hole AUC 0.56–0.58 (PS), 0.60–0.67 (TM) vs our map 0.91 / 0.88. Re-embedding does flip the decoder (score 9.0 → −10.5) | none | — | — | — | **fail** |
| 11 | Known-message CAM / per-position logits | all | Repo: already tried, AUC 0.61–0.64 full. PixelSeal's map is 8×8 before pooling (videoseal `pixel_decoder.py`) | none | — | — | — | failed earlier |
| 12 | WAM-style network | all | Repo: CV 0.450. markbench: WAM IoU 0.05–0.36 by region size, degrades when 95 % is watermarked | none | — | — | — | failed earlier |

Pass thresholds were written in each script's docstring before running.

## 3. Top 3 in detail

### 3.1 Zoomed inference for rectangles

**Mechanism.** In an outpainted image the original photo, already watermarked, was shrunk into a small rectangle. The watermark texture shrank with it by a factor of roughly 3 to 6. Our segmenter learned what the watermark looks like at the scale of full images, so at native resolution a shrunk watermark looks like nothing. Cutting the image into tiles of side 512/z and resizing each tile to 512 brings the texture back to a scale the network knows. The same magnification makes ordinary (unshrunk) watermarks too coarse, which is why zoom ruins the object and full maps.

**Numbers [measured], folds 3–4 held out, one quick model per method:**

| | zoom 1 | zoom 3 | mean of zooms 1–6 |
|---|---|---|---|
| TM rect AUC / top-5 peak hit | 0.68 / 0.44 | 0.91 / 0.83 | 0.94 / 0.89 |
| SS rect AUC / top-5 peak hit | 0.61 / 0.13 | 0.81 / 0.53 | 0.82 / 0.53 |
| TM object AUC | 0.94 | 0.59 | 0.86 |
| SS full AUC | 0.91 | 0.71 | 0.87 |

PixelSeal was not run (time). The network was never trained on zoomed tiles, so these are lower bounds.

**Integration.**
1. `e2e.py`, `predict()`: add the tiling function `zoom_map` from `research/tests/test_zoom.py` and write a second set of maps, e.g. tag `e2eZ_<method>` = mean over zooms 2, 3, 4 (keep the zoom-1 maps as they are).
2. `e2e.py`, `batches()`: add scale augmentation — with some probability take a random crop of side 128–256 px and resize it to 512, with its mask. This teaches the shrunk-watermark scale directly. `synth.py` `outpaint()` already produces shrunk rectangles; check that their shrink factors match the real ones (real rectangles: median side 60–100 px).
3. `seedrect.py`, `seeds()`: take seeds from the top-5 peaks of the `e2eZ` map instead of thresholded components of the zoom-1 map. Stage 1: shifts in 4 px steps (not 8) and scale factors 1.0, 1.1, 1.2 only, because a box 5 % too large keeps 99 % of the StegaStamp score and a box 10 % too small keeps 28 %.
4. `make_sub2.py`: when no box is accepted on a gate-partial image, try `research/tests/test_maxrect.py`'s max-sum rectangle on the `e2eZ` map before falling back to the threshold.
5. Rectangle vs object routing: an accepted box means rectangle. Otherwise compare the zoom-1 and zoom-3 map maxima (objects light up at zoom 1, rectangles at zoom 3). This rule is untested.

### 3.2 SAM prompted by our map, for transplanted objects

**Mechanism.** On transplanted objects our map is right about where the watermark is (AUC 0.94–0.98) but its outline is blurry. The watermarked region is a visible object, and SAM draws object outlines well. We click SAM at the brightest point of our map and take the mask it returns that has the strongest contrast in our map. If that mask barely overlaps our own thresholded blob, we keep our blob.

**Numbers [measured], all partial-object train images, true regime:** PS 0.643 → 0.730, TM 0.682 → 0.793, SS 0.708 → 0.804 with the `agree` rule (SAM used on 63–83 % of images). The best of SAM's three masks would give 0.82–0.84, so a better selection rule has another 0.03–0.08. The baseline is a threshold fitted in-sample, without the `largest` post-processing; against the real pipeline (0.66 / 0.71 / 0.72) the gain is +0.07 to +0.09.

It does not work on full images: holes are not visible objects for SAM (best of 3 masks 0.52–0.57, below the 0.59–0.67 baseline). That agrees with the docstring saying holes are restored clean-original pixels.

**Integration.** `make_sub2.py`, `method_preds()`: after `postproc.apply_all`, for each test image the gate calls partial and that got no box, run the `pick` + `agree` logic of `research/tests/test_samprompt.py` and replace the mask. SAM ViT-B needs about 0.1 s per image. Fit the agreement cut-off (0.3) and the ring width in CV with `recteval.py`'s fold loop first.

### 3.3 Full regime: specialised model and faithful holes (untested)

**Mechanism.** Half the score is here and nothing tested tonight moves it. What we know: the map ranks pixels well (AUC 0.90–0.93) but 58 full images miss the hole and 117 get its shape wrong; no image-content cue exists (TruFor at chance, SAM fails), so the watermark signal is the only information. One network currently serves three very different problems at three watermark scales. A model trained only on hole-type data, at the full-image watermark scale, with a 128×128 output, spends all its capacity on thin and small holes.

**Integration.**
1. `e2e.py`: add `--kinds holes` to filter the synthetic shards by `meta.json` `kind`, fine-tune on real full images only, `--out 128 --res 512`. Use it only where the gate says full.
2. `synth.py`, `holes()`: make "restored to the clean original" (`out[hole] = a[hole]`) the only variant, and sample hole areas from the real distribution (median 6 %, a quarter under 2.3 %, 2 components).
3. Decide on CV full-regime mIoU against `ens*` (0.595 / 0.609 / 0.673). Stop if the gain is under 0.02.

## 4. Answers

**How was the data generated?** Unknown. The Hugging Face repos contain no generation script and the PNGs no metadata (checked). Facts: `wmloc/data.py` says full images had "a part restored to the clean original"; train is exactly 85 images per method × regime. The September 2026 benchmark ("What Breaks Local Watermarks?", arXiv 2609.16832) is **not** from SprintML (Edinburgh / Garandor), has no StegaStamp and uses only SD 1.5 inpainting, so the task was not built from it. Its local-embedding recipes do match the task text: TrustMark = crop the box, resize to native resolution, add the residual back; PixelSeal = embed globally, keep the residual inside the mask.
Implication for synthetic data: composite first, round-trip afterwards (otherwise the synthetic images carry a seam the real ones lack — **[measured]** no seam at real rectangle borders); fill holes with the clean original; match rectangle shrink factors.

**What has a 0.78 team found?** Reasoning, not evidence. By the arithmetic above they need about 0.70 on full and 0.85 on partial. Partial at 0.85 is reachable with what passed tonight (scale-aware rectangles, object outlines). For full they most likely have a sharper segmenter trained on a closer reproduction of the generation pipeline, possibly a pixel-space network at full resolution as in MaskMark, rather than a different kind of signal: every other signal we or the literature tried is far below a learned map.

**Working assumptions.**
1. *VAE round trip erases pixel-exact patterns but keeps the watermark decodable* — true for FLUX on our synthetic (`vae_test.py` log: decode 1.00 for all three, matched filter 0.02–0.08); false for the SDXL VAE (PixelSeal decode 0.55). Whether the real images went through a FLUX VAE is **unknown**: their second-round-trip change is 1.9–2.3/255 against 3.0–3.6 for a first trip on COCO (`test_vae_level.py`), but we have no reference value for a true FLUX output. No published FLUX numbers were found.
2. *Generative edits leave traces that survive a full-image round trip* — false for off-the-shelf detectors (TGIF 2024: TruFor F1 0.83 → 0.19; ours: AUC 0.44–0.50), true for detectors fine-tuned on the matching generator (TGIF2 2026: 0.76–0.94). Probably irrelevant here if holes are clean originals.
3. *These decoders are crop-robust* — false for StegaStamp (4 px needle; WAVES: resized-crop TPR 0.54), mildly true for TrustMark (trained with scale 0.8–1.0 and a 244/256 crop; our search basin is wide), partly true for PixelSeal (paper Fig. 8: bit accuracy 0.58 to 0.98 depending on crop size).
4. *An extra 2× downscale kills watermarks in small rectangles* — supported: magnifying 3× restores the signal (row 1), and 512 input beat 256 input on partial earlier (AUC 0.79 → 0.88, HANDOFF).

## 5. References

- What Breaks Local Watermarks? arXiv 2609.16832 (2026) — https://arxiv.org/html/2609.16832 ; code https://github.com/garandorinc/markbench
- MaskMark, arXiv 2504.12739 (2025) — https://arxiv.org/html/2504.12739v2 ; https://github.com/hurunyi/MaskMark
- Watermark Anything, arXiv 2411.07231 (ICLR 2025) — https://arxiv.org/abs/2411.07231
- PixelSeal, arXiv 2512.16874 — https://arxiv.org/abs/2512.16874 ; videoseal `pixel_decoder.py` https://raw.githubusercontent.com/facebookresearch/videoseal/main/videoseal/modules/pixel_decoder.py
- TrustMark, arXiv 2311.18297 — https://arxiv.org/html/2311.18297
- WAVES (StegaStamp robustness), arXiv 2401.08573 — https://arxiv.org/html/2401.08573
- SyncSeal, arXiv 2509.15208 — https://arxiv.org/html/2509.15208v1
- TGIF (2024) and TGIF2, arXiv 2603.28613 (2026); DinoLizer, arXiv 2511.20722 — https://github.com/anonyme610/dinolizer
- SAFIRE (AAAI 2025); TruFor — https://github.com/grip-unina/TruFor
- Extremal perturbations, arXiv 1910.08485; Lovász hinge, arXiv 1705.08790

Caveat on sources: paper numbers were read through a summarising fetch tool by research sub-agents; spot-check before quoting them elsewhere. Several papers on the brief's list (EditGuard, OmniGuard, StableGuard, GIM, BR-Gen and others) were not read in the time available.
