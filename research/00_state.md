# State at Sun 22 Nov 2026, ~00:15 (written before the research)

- **Best submission:** `results/sub_step4f_psbox.pt`, CV 0.653, public LB 0.6456 (id 1832, Sat 22:52). Leaders: 0.78 (per the brief; HANDOFF still says 0.72).
- **Pipeline:** `vae_synth.py` (COCO synthetic edits + FLUX-VAE round trip) → `e2e.py --res 512` (decoder early layers + head, pretrain synthetic, fine-tune per fold) → ensembles `ens3_pixelseal`, `ens2_trustmark`, `ens2_stegastamp` → `bruterect.py` boxes (PixelSeal, TrustMark) → `make_sub2.py` (gate + `postproc.py` per regime).
- **Running:** one chain on GPU 0 of jrc0423 (`amelio`: StegaStamp/PixelSeal centred box search, then `e2e --out 128 --foldens`, then seeds 1-3). 3 GPUs free. Node ends ~11:17 Sun; reservation ends 14:00. Compute nodes have no internet; scratch inode quota is nearly full (87k left).

CV mIoU of the current best, method × regime (85 images per cell; `research/tests/state_table.py`):

| method | regime | map + post | + boxes | minority AUC (median) | points lost (of 1.0) |
|---|---|---|---|---|---|
| PixelSeal | full | 0.595 | 0.595 | 0.899 | 0.068 |
| PixelSeal | partial | 0.577 | 0.648 | 0.761 | 0.059 |
| TrustMark | full | 0.609 | 0.606 | 0.901 | 0.066 |
| TrustMark | partial | 0.630 | 0.790 | 0.883 | 0.035 |
| StegaStamp | full | 0.673 | 0.669 | 0.932 | 0.054 |
| StegaStamp | partial | 0.604 | 0.610 | 0.828 | 0.066 |

- **Where the points are:** the three full cells lose 0.188 together, the partial cells 0.160. To reach 0.80 every cell must average 0.80; full is at 0.60–0.67.
- **Partial is two different problems** (`test_seedhit.py`): transplanted objects have map AUC 0.94–0.98 but mIoU only 0.66–0.72 (shape precision); outpainting rectangles have AUC 0.46–0.75 and mIoU ≈ 0.49 without boxes (the map does not see them at all).
- **Full:** AUC 0.90–0.93 yet mIoU 0.60–0.67. `diag_full.py`: 58 of 255 images miss the hole entirely (score 0.48), 117 have a bad shape (0.57–0.62), 80 are fine (0.75–0.80).

Contradictions between the brief and the repo:
- Brief: "StegaStamp crop decoding 0 %, search is flat inside R". Repo (`amelio` log, `diag2a.py`): the true box beats the found box in 66 % of StegaStamp rectangles and the score peak is ~4 px wide (8 px shift → 3 % of the score). The search fails; the signal is there. HANDOFF: exact box decodes for 64 % (SS), 84 % (TM), 33 % (PS).
- Brief: probe AUC "0.86 full / 0.77 partial". Current ensembles are higher on full (0.90–0.93).
- Brief: e2e TrustMark 0.581. Current TrustMark is 0.698 with boxes.
