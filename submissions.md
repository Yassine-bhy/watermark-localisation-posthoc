# Submissions log

| date | file | method / settings | local CV mIoU (train) | leaderboard |
|---|---|---|---|---|
| 2026-11-21 | results/sub_step0_allzeros.pt | constant all-0 (format check only) | 0.2518 (no fitting, whole train) | not submitted |
| 2026-11-21 | results/sub_step1_signals.pt | per method best signal + full/partial gate (PS agree_z, TM winmean, SS resid), thresholds fit on train | 0.4733 (5-fold CV; PS 0.470 / TM 0.465 / SS 0.485) | 0.4695 (public, id 1178) |
| 2026-11-21 | results/sub_step3_probe.pt | decoder-feature probes (PS, TM: avg of early-layer + deep-layer heads; SS: conv2/4/5 head) + full/partial gate, per-regime thresholds | 0.5311 (5-fold CV; PS 0.556 / TM 0.544 / SS 0.494) | 0.5307 (public, id 1438) |
