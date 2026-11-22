---
name: ml-competition-playbook
description: Use at the START of any ML competition, Kaggle challenge or hackathon (leaderboard score, train/test split), and again whenever progress stalls or the leaderboard top is far ahead. A playbook distilled from a lost race (watermark-localisation hackathon, Nov 2026: we scored 0.739, two teams with textbook recipes scored 0.80–0.81). Forces a standard strong baseline first, augmentation ablations instead of theorising, measuring the real data instead of guessing, honest CV, scale and ensembles, and disciplined time/compute use.
---

# ML competition playbook

## Why this skill exists (what went wrong, Nov 2026)

Task: segment the watermarked area of images (3 watermark methods, 510 labelled train images, 1020 test images).
We finished 0.739; two teams reached 0.80–0.81 — one of them with no ML background, just a model following the
standard recipe. Both winners used the same textbook ingredients. We lost the gap for these reasons:

1. **No standard strong baseline.** We started from a frozen-feature probe + tiny head and only tweaked it. The
   winners built a **full U-Net / FPN with a pretrained encoder** (the domain's own pretrained network) from the start.
   (One model: 0.61 → 0.67 from the architecture alone.)
2. **Augmentation reasoned away instead of tested.** We argued that flips / resizing "break the signal". The single
   most standard augmentation — **random resized crop (scale ~0.9–1.1) + small shifts** — matched how the real data
   had been processed (a ~1.03 zoom after embedding): PixelSeal 0.71 → 0.75 for one team; for the other, a model
   trained on synthetic data only went from 0.07 to 0.77 on real images once the zoom was added. One 20-minute
   ablation would have found it.
3. **Guessed the data-generating process instead of measuring it.** We assumed a VAE round trip and built on it all
   night. The winner *measured* the transform directly from the pixels (interpolation artefacts, percentiles 0.986 /
   1.034 / 1.065) and reproduced it in the synthetic data.
4. **Hours on exotic ideas before the basics worked** (hand-made signal maps, attribution, forensic detectors, matched
   filters, DINO features) — all ≈ chance. Winners had 0.66 after ~1 h and 0.70 after ~2 h; we reached 0.70 after ~12 h.
5. **Reactive micro-management** (status updates, scheduling, quota fixes) instead of stepping back to ask
   "how far are we from a strong standard pipeline?".
6. **Synthetic-data realism found late.** Building synthetic training data from the competition's own images
   (real backgrounds, real mask shapes) was our biggest gain — but it came at 01:00, not in the first hours.

What we did right (keep doing it): honest CV that predicted the public score within ±0.006; kill tests with
thresholds written before running; a journal of refuted ideas; research agents in parallel; scaling data
(+0.01 per doubling); decoder/verifier-based post-processing where the problem has an exact verifier; SAM snapping
for object-shaped targets.

## Phase 0 — first 30 minutes: understand and measure (no modelling yet)

- Read the task text, the starter kit, the data loader **docstrings and comments** (they often describe how the data
  was generated). Write down the metric exactly and implement it locally; check it against the official one.
- Look at 30+ examples per class / category with labels. Split the data into the **categories that the metric treats
  differently** (e.g. small vs large regions, regimes, sources) and count them. Note what a constant prediction scores.
- **Measure the real data's processing**, do not guess it: resolution / interpolation artefacts, JPEG grids,
  noise level, colour statistics, object sizes, positions, border contact — and compare with any synthetic data you
  plan to generate. Write the comparison table before training on synthetic data.
- Check for leakage and structure quickly: duplicates (thumbnail correlation, SIFT), ID patterns, train/test overlap.

## Phase 1 — first 2–3 hours: the standard strong baseline

Use the textbook recipe for the task type before any custom idea:

| Task type | Baseline |
|---|---|
| Segmentation | U-Net / FPN with a **pretrained encoder** (domain network if one exists — e.g. the official detector/decoder without its head; else ImageNet ResNet/ConvNeXt), multi-scale input if objects vary in size, BCE + Dice (per image) |
| Classification / detection on images | pretrained CNN/ViT fine-tuned, standard detector (e.g. YOLO-family / DETR) |
| Tabular | LightGBM / CatBoost with sensible CV, then feature work |
| NLP | pretrained transformer fine-tuning |

- **Augmentations: ablate, never theorise.** For each standard augmentation (random resized crop with small and
  medium scale ranges, shifts, flips, rotation, colour, blur, JPEG, noise) run one short training with and without it
  and keep what helps on CV. Domain arguments ("this would destroy the signal") are hypotheses to test, not reasons to skip.
- **Mix real and synthetic in every batch** (e.g. 20 % real / 80 % synthetic) rather than pretrain-then-fine-tune only.
- Train long enough (thousands to tens of thousands of steps); log CV per category.
- Submit the baseline early to calibrate CV against the public leaderboard.

## Phase 2 — make synthetic / training data match the real data

- Generate training data from the competition's **own** images and label shapes whenever possible (real backgrounds,
  real mask outlines, SAM segments of real images as shapes).
- Validate realism by **zero-shot transfer**: a model trained only on synthetic data, scored on real labelled images.
  Optimise that number first; then scale the data (measure the gain per doubling).

## Phase 3 — validation discipline

- K-fold CV on the real labelled data, same folds everywhere; every threshold, routing rule and model choice fit on
  training folds only; CV the **whole pipeline**, not components alone. Use nested CV when choosing among many options.
- Prefer a small, fixed candidate set decided in advance; changes under ~0.005 are noise on small datasets.
- Never train on pseudo-labels of the test set without checking contamination (a model can memorise its teacher on
  the test images — local CV improves, leaderboard drops).
- The public leaderboard is a check, not an objective; the private split decides.

## Phase 4 — scale, ensemble, post-process

- Many seeds and larger data/models; average logits; **shift TTA** (and flips/rotations if the ablation allows).
- Per-category thresholds; a learned selector (e.g. LightGBM over candidate masks/predictions with per-image features)
  can beat hand rules.
- Exploit exact verifiers when the problem has them (decoders, checksums, physics): use them to accept/reject
  candidates, calibrated on train.
- For object-shaped targets: snap predictions to SAM segments when they agree.

## Process rules (throughout)

- Keep a journal: idea → kill test (threshold written first) → result → verdict. Never retry a refuted idea without a
  new reason. Read it before proposing anything.
- Every 2–3 hours, or when 3 tweaks in a row give < noise: stop and compare the pipeline with a strong standard one;
  re-read the leaderboard and ask "what simple thing could the top teams have found?" (data recipe, augmentation,
  architecture, leakage, metric quirk) — check each in minutes.
- Run a research agent / literature search in parallel from the start, with a sharp brief (measured facts, failures,
  the exact question).
- Compute: plan runs so they finish before the deadline and job time limits; one job per GPU unless the GPU is idle
  (sharing a busy GPU halves both); write few large files (inode quotas); verify outputs exist before using them;
  give an expected duration for every run.
- Before publishing code: no API keys / tokens in history, no large binaries (> 100 MB) — rewriting history removes
  formerly tracked files from disk on checkout, so back them up first.
