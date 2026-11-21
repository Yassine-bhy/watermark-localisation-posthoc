# Submissions log

| date | file | method / settings | local CV mIoU (train) | leaderboard |
|---|---|---|---|---|
| 2026-11-21 | results/sub_step0_allzeros.pt | constant all-0 (format check only) | 0.2518 (no fitting, whole train) | not submitted |
| 2026-11-21 | results/sub_step1_signals.pt | per method best signal + full/partial gate (PS agree_z, TM winmean, SS resid), thresholds fit on train | 0.4733 (5-fold CV; PS 0.470 / TM 0.465 / SS 0.485) | 0.4695 (public, id 1178) |
| 2026-11-21 | results/sub_step3_probe.pt | decoder-feature probes (PS, TM: avg of early-layer + deep-layer heads; SS: conv2/4/5 head) + full/partial gate, per-regime thresholds | 0.5311 (5-fold CV; PS 0.556 / TM 0.544 / SS 0.494) | 0.5307 (public, id 1438) |
| 2026-11-21 | results/sub_step4b_ens.pt | PS: avg(step-3 probe, end-to-end decoder segmenter pretrained on FLUX-VAE synthetic data); TM step-3 probe; SS probe on all dec_conv layers; per-regime post-processing (smoothing + fill / largest component) | 0.5510 (5-fold CV; PS 0.577 / TM 0.548 / SS 0.528) | 0.5569 (public, id 1645) |
| 2026-11-21 | results/sub_step4e_r512.pt | e2e segmenters on FLUX synthetic data; TM+SS: avg of 256-input and 512-input models; TM: brute-force rectangles on gate-partial images (score > tau); PS: avg(probe, e2e 256); per-regime post-processing | 0.638 (5-fold CV; PS 0.577 / TM 0.698 / SS 0.639) | 0.6323 (public, id 1742) |
