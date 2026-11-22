# REPORT_3 — research round 3 (Sun 22 Nov 2026, 02:15–02:40), stopped early at the user's request

## Verdict
No shortcut found. The leaders' edge is most likely more and better-matched training data, which is the user's call.
Next step: scale up RS data (real-domain synthetic, see REPORT_FULL.md): more images, several data seeds, all methods.

## Kill tests run (CPU, same 5 folds)
| idea | test | result | verdict |
|---|---|---|---|
| Leakage: images share source content at another scale/position (outpaint = shrunk copy of another image, transplanted object from another image) | `tests/test_siftdup.py`: RootSIFT over all 1530 images, RANSAC similarity; pass = ≥ 5 % of images with a ≥ 25-inlier partner | 0 pairs ≥ 25 inliers (max 12) **[measured]** | KILLED |
| Missed holes caused by flat regions (PixelSeal JND attenuates the watermark there) | `tests/eda_texture.py`: hole-vs-wm AUC by texture tercile | Opposite: **textured** holes are missed. Mean map value in holes, low→high texture: PS 0.46/0.51/0.58, TM 0.35/0.41/0.41, SS 0.25/0.31/0.41; AUC falls with texture **[measured]** | flat-region idea dead; textured holes are where the loss is |
| Edge-aware refinement (guided filter, image as guide) before postproc | `tests/test_guided.py`; pass = one (r, eps) gives ≥ +0.010 mean full CV, no method < −0.003 | full CV: PS 0.595 → best 0.595 (r=4, eps=0.01), TM 0.600 → 0.609 (best of 9 settings, selection bias); SS not finished **[measured]** | FAIL (PS flat, TM gain is within noise) |

## Web research (two agents; notes in `research_notes/r3_competition.md`, `r3_literature.md`)
- No organiser paper, generation script or public solution for this image task. HF repo: 2 commits, "v2.1", no pipeline hints. [found]
- Previous SprintML hackathon (text watermark localisation, June 2026), rank-2 write-up: fuse all detectors, strong length/shape prior, only ship changes that hold on several validation splits. https://github.com/florianzainzinger1/cispa-hackathon-grand-finale-2026
- Internals: TrustMark residual made at 256, mean-subtracted, upsampled bilinearly; PixelSeal residual made at 256 × full-res JND map. Our texture EDA shows the missed holes are textured, so this does not explain them.
- Literature: nothing published localises global watermarks at pixel level beyond WAM / MaskWM (arXiv 2504.12739), whose gains come from large synthetic training sets. Expected-IoU thresholds, box search and SAM are already in our journal.

## Recommendation (user's direction): more data, train on it
1. RS data at scale: `realsynth_prod.py <method> post <steps> 3000 <tag> <seed>` with 2–3 data seeds per method (vary bases × holes × messages), all three methods. Add the method's own train masks for the final test-time model.
2. Ensemble the RS models with the current maps on gate-full images (`make_ensrs.py`); then run `pipecv.py`, submit if CV gain > 0.005.
3. Bias the hole sampling toward textured regions (that is where holes are missed).

## Search log (do not repeat)
SprintML/CISPA/organiser papers; HF dataset files and commits; earlier SprintML hackathons; NeurIPS 2024 Erasing the Invisible; markbench 2609.16832; MaskWM 2504.12739; WAM; OmniGuard/EditGuard; block-level 2512.14994; TrustMark trustmark.py; PixelSeal embed path; ForgeryTTT; Nowozin expected-IoU; self-training (2006.06882, 2004.14960).
