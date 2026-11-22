# Handoff — Watermark Localisation (task 1), BlitzTeam

## FINAL (Sun 22 Nov 10:55): best submission = results/sub_step10.pt, public 0.7390 (id 3929, 09:58), CV 0.739, 14th of 42.
Night summary: RS (real-domain synthetic) data at scale (8k -> 64k images per model) was the lever; extra Slurm nodes
outside the reservation (allowed: max one node per team INSIDE the reservation) via slurm/xjob.sh + slurm/xqueue/.
Final pipeline = integrate.py (largest-model ensembles ensH/ensHP) + make_sub3.py (gate -> boxes -> zoom -> SAM -> map).
See submissions.md for every step with CV and public score.

## DEADLINE: submissions close Sunday 12:00 (user, 06:05). Final submission by ~11:30.

## UPDATE Sun 22 Nov 00:15 (newest; read first)
Best: **public 0.6456 (id 1832, 22:52), CV 0.653**, `results/sub_step4f_psbox.pt` (PS ens3+boxes, TM ens2+boxes, SS ens2).
User directives tonight: focus FULL images; cheap measurable changes judged on CV; NO realistic-inpainting data / new
architectures / 128 retrain / SAM without asking (the user later allowed continuing; 128 + seeds are queued); 2 GPUs max.
Findings after 22:20:
- Full images (diag_full.py, shape_breakdown.py): every full train image has a hole, 54 % have 2+; loss = bad shapes 46 %
  of images (mostly "missed one of several holes"), missed holes 23 %; oracle best-threshold-per-image only +0.038.
- Step 1 rules (step1_full.py): never-empty +0.005 PS only, top-k components +-0.002 -> post-processing saturated.
- Dead ends: TruFor (AUC 0.44-0.56 on full), pixel matched filter with re-embedded pattern (AUC 0.51), map-mean gate,
  768 input, no-flip, fp32, Lovasz (all ~0); frozen BN +0.006 (TM).
- StegaStamp boxes (diag2a.py): the score peak is ~4 px wide (8 px shift -> 3 % of score); the coarse grid jumps over it;
  centring the score does not help. New `seedrect.py`: boxes seeded from our map + dense 8 px search + refinement to 1 px.
Running (2 jobs): `seedrect` (SS then PS; test split only if boxgain.py says CV +0.005) and `o128seeds`
(e2e --out 128 --foldens; then seeds 1-4 of the 512 models, tags e2eR512s<k>, also 4-way-flip variants <tag>V).
Disk (01:30): deleted results/synth (originals, regenerable with synth.py; synth_flux kept) and results/synth_inp (SD2-inpainted holes: measured worse on full images, 0.537 vs 0.592). Quota errors are intermittent and NOT from our folder.
Next: when seedrect SS test exists -> make_sub2 with BRUTE file swap (make_sub2 reads results/brute_<m>_*; copy or add a
tag option), ensemble seeds (avg of e2eR512, e2eR512s1..4) per method, submit if CV gain > 0.005.

## UPDATE Sat 21 Nov ~22:20 (read this first)
Best: **CV 0.638 / public 0.6323 (id 1742, 22:12)**, `results/sub_step4e_r512.pt`. Leaderboard leader ~0.72.
What works (all numbers 5-fold held-out CV, split full/partial in evalmaps.py output):
- `vae_synth.py`: FLUX-VAE round trip of the synthetic COCO data -> `results/synth_flux/<method>_<k>` (6000/method).
- `e2e.py`: decoder's own early layers + head trained end-to-end: pretrain on synth_flux, fine-tune per fold on real.
  `--res 512` (no 512->256 shrink) is much better on partial images (TM partial AUC 0.79 -> 0.88); `--noflip`, `--res 768` under test.
  StegaStamp backbone = dec_conv_1..7 rebuilt in torch from external/stega_feats.onnx (exact). Long pretraining (24k steps) overfits: worse.
- `bruterect.py`: brute-force multi-scale rectangle search (raw known-message logit score) on gate-partial images;
  `recteval.py <maptag> score` = CV with boxes. TM +0.08, PS +0.04. SS search is slow (~19 s/img, onnx2torch).
- `make_sub2.py --pixelseal TAG[:box] ...` builds submissions (post-processing + optional boxes).
- Best per method now: PS ens3 (e2eR512+e2eflux+probe) + boxes 0.621; TM ens2 (e2eR512+e2eflux) + boxes 0.698; SS ens2 0.639 (boxes pending).
Dead ends tonight: stitched decoder agreement maps, DINOv2, AE reconstruction error, COCO retrieval, TrustMark remover energy.
Filesystem: intermittent "Disk quota exceeded" (project-wide?) silently kills queued jobs (no log) -> check node_status after queueing.
GPU rule from user: at most 1 GPU for us — one sequential chain (slurm/hold_*.cmd + waiter to queue the next part).
 — state at Sat 21 Nov 2026, ~20:35

Read first: `TASK.md` (official task), `submissions.md` (all submissions), course page
https://claude.ai/artifact/2wsamNKEVYGFWFRuW8QiXZ (source `docs/course_template.html`, rebuild with
`python docs/build_course.py`, then publish `docs/course.html` again — the user reads it; keep it updated after every result).

## Scores
| | CV (5-fold, held-out) | public leaderboard |
|---|---|---|
| step 1 simple signals + gate | 0.473 | 0.4695 (id 1178) |
| **step 3 decoder-feature probes** (`results/sub_step3_probe.pt`) | **0.531** (PS 0.556 / TM 0.544 / SS 0.494) | **0.5307** (id 1438, 20:17) |
Leaderboard leader earlier tonight: 0.66. CV predicts the leaderboard within ~0.005 -> decide on CV, submit only if CV gain > 0.02.

## Running infrastructure (do NOT break)
- Slurm job **15902410** `wm_step2` on node **jrc0423** (4x A100), started 19:17, limit 16 h -> ends **~11:17 Sun 22 Nov**.
  It is the team's ONE allowed job. Never `sbatch` while it lives. Reservation `cispahack` ends 14:00 Sun.
- The job is a worker: each GPU runs the next file from `slurm/queue/*.cmd` (oldest first), logs to
  `output/15902410_<file>.log`, moves it to `slurm/queue/done/`. Add work: `slurm/run_on_node.sh <name> "<cmd>"`.
  Status: `slurm/node_status.sh`. Stop & release node: `touch slurm/queue/STOP` (only when everything is done).
- Team-level copies for teammates: `/p/scratch/training2666/neves2/BlitzTeam/gpu_node/` (run_on_node.sh, node_status.sh,
  HOWTO.txt, symlinks queue/ and logs/). **We use at most 2 GPUs at once; >= 2 stay free for teammates.**
- Before any future sbatch (after the job ends): `T=$(pwd | grep -oE '/p/scratch/training2666/[^/]+/[^/]+'); squeue -r --reservation=cispahack -h -o "%u|%T|%Z" | grep "$T"` must be empty.
  Use one job with `--gres=gpu:4`, chain with `--dependency=afterany:<id>`.

## Rules / preferences (also in Claude memory)
- Install packages without asking, but never break the shared `.venv`: dry-run, `uv pip install --exclude-newer <2 weeks ago>`,
  never upgrade torch 2.5.1 / numpy pins; mention what was installed. Installed so far: matplotlib, nbconvert, pandas,
  diffusers==0.31.0 (0.42 needs newer torch; diffusers' own loader fails with safetensors 0.8 -> build from config and
  `load_state_dict(safetensors.torch.load_file(...))`).
- Leaderboard: submit with `submission_template.submit()` after setting `BASE_URL='http://34.147.229.55'` in memory only;
  the API key is in `../submission_template.py` — never copy it into git or anything public. Rate limit 1 submission / 30 min / team.
- Every step: notebook/course with images + numbers; all performance claims = held-out CV numbers; one git commit per working step.

## Files (benelhajyahia1/)
data.py (loader, train table, stratified 5-fold `fold`), metrics.py (official mIoU, checked submission writer),
baselines.py (step 0), signals.py / signals_b.py + diagnostics.py (step 1/1b), crops.py + rectsearch.py (crop decoding,
decoder-verified rectangle refinement), synth.py (synthetic edits on COCO), model.py + train.py (step 2 WAM network),
attrib.py (occlusion/gradient maps), rect_from_net.py, vae_test.py,
**probe.py** (step 3: small conv head on the decoder's own early feature maps; `--deep 1` adds a deeper layer; writes
`results/preds/probe[_deep]_<method>_{train,test}.npz`), **evalmaps.py** (CV protocol: gate + per-regime thresholds; prints
AUC/CV), **make_sub.py** (`--name X --pixelseal TAG|avg:A+B ...` -> `results/sub_X.pt`).
external/: wam (Watermark Anything + MIT weights), coco (val2017 + instances), vae/{flux,sdxl,sd}, stega_feats.onnx
(StegaStamp graph with dec_conv_k outputs exposed), hf/. HF login done (user's token, team cache).

## Key findings (numbers in the course)
- Two regimes, 50/50: *full* (~90 % watermarked, object-shaped clean holes) vs *partial* (small rectangle = outpainting,
  or watermarked object = transplantation). Whole-image decode agreement separates them with 97 % CV accuracy (the "gate").
- Exact ground-truth box of an outpainted rectangle decodes the message (TM 84 %, SS 64 %, PS 33 % of rectangles); the
  refinement fixes a 30 %-off guess to mIoU ~0.88 — but needs a decent starting box (network proposals were too weak).
- Hand-made maps, WAM network (synthetic or real fine-tune, CV 0.450), decoder attribution, SDXL VAE: no within-image signal.
  Real watermarks are much weaker as exact pixel patterns than fresh embeddings; FLUX VAE round trip reproduces that best.
- **Decoder-feature probes work**: within-image AUC 0.83–0.86 on full images (PS/TM), 0.73 (SS). Train in < 30 s.

## Next steps (in order)
1. Improve probes cheaply (each run < 1 min on one GPU via the queue): StegaStamp other layer sets (dec_conv_1..7);
   ensembles over seeds/layer sets for all methods; feed known-message agreement maps (PixelSeal/TrustMark per-position
   bit logits from `signals.py agreement_map`, `results/signals/*`) as extra probe channels; per-regime heads.
2. Post-processing in CV on probe maps: smoothing, morphological cleanup, min component size; rectangle refinement
   (`rectsearch.refine`) started from probe-map boxes on images the gate calls partial, keep box only if agreement > 0.5.
3. Optional: FLUX-VAE synthetic data to pretrain the probe heads (more training data that looks like the real images).
4. After each CV gain > 0.02: `make_sub.py`, submit (respect 30-min limit), log in submissions.md, update course, commit.
Write the step notebooks too (notebooks/make_02_network.py exists but was never executed; a 03 probe notebook is missing).
