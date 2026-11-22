# R3 literature notes: improving mask quality (2026-11-22)

## Query log
1. "What Breaks Local Watermarks" arXiv -> 2609.16832 (HTML read)
2. WAM 2411.07231 mIoU -> HTML read
3. PixelSeal 2512.16874 embedder resolution JND -> HTML read
4. extremal perturbation mask optimisation watermark decoder known message -> MaskMark/MaskWM 2504.12739 (read), no direct hit for decoder-in-loop mask optimisation
5. (fetch) "Where is the Watermark?" 2512.14994 block-level (MELB): own keyed DWT scheme, not applicable
6. TrustMark residual 256 upsample / detector -> read adobe/trustmark python/trustmark/trustmark.py
7. videoseal/models/videoseal.py (PixelSeal embed path) read
8. OmniGuard/EditGuard tamper localization -> 2412.01615 read
9. forgery localization CRF / guided filter / TTA -> ForgeryTTT 2410.04032; no CRF/TTA IoU numbers found
10. TrustMark DETECTFIRST localize detector -> no docs; 2611.02030 is LLM-text (irrelevant)
11. Nowozin expected-IoU decision (CVPR 2014)
12. self-training segmentation (Zoph 2006.06882; Zhu 2004.14960)
13. sliding-window decoding global watermark localization -> nothing beyond WAM/MaskWM

## Internals facts (d)
- TrustMark (trustmark.py): cover resized to 256 (bilinear), residual = stego-cover, per-channel mean subtracted, bilinear-upsampled to full res, x WM_STRENGTH (P variant x1.25, forced centre-square crop); decoder input 245 (Q/C/B) or 224 (P). => carrier is smooth at >=2 px at 512; 4-px output cells are fine; aspect>2 crops centre only.
- PixelSeal: embedder 256x256, residual upsampled, then multiplied by JND map computed at full res: x_w = x + a*(up(w) * JND(x)). JND small in flat areas, large at edges; paper: watermark "concentrates near object edges". Ablation: no-JND PSNR 43.1 vs 47.0. Decoder ConvNeXt-v2-T at 256. Crop robustness: keep 0.55 -> 0.90 bit acc, 0.32 -> 0.58.
  => PixelSeal holes in flat regions carry ~no evidence either way; per-pixel net must interpolate from textured neighbours (edge-aware propagation helps).
- 2609.16832 benchmark: PixelSeal "provides no spatial output"; TrustMark localised only via its own box detector; small regions (0-3% area) hardest (MaskWM IoU 0.634 vs 0.825 for 40-85%; WAM 0.052).
- WAM: ViT+pixel decoder at 256, BCE detection; DBSCAN min 2% area. MaskWM-D (global embed, decoder localises with U2-Net, mask types full/rect/irregular/COCO segments, LaMa-style) = closest analogue to our setup; beats WAM.
- OmniGuard: edge loss weight 20 for boundaries; IoU 0.93 clean.

## Ranked ideas
see final report (same content).
1. Metric-aware post-processing on OOF probs (Nowozin expected-IoU per-image threshold, min-component removal, keep-k-components, rectangle fit for outpaint partials). 0 GPU. Kill: OOF CV delta in 15 min.
2. Known-message box search for outpainted rectangles: crop candidate box from predicted bbox +-jitter, resize to decoder res, score bit agreement / logit margin with known message; pick argmax. Kill: 50 train rect-partials, IoU vs net mask.
3. Test-image pseudo-label self-training (+0.9..1.5 mIoU in lit).
4. Edge-aware refinement (guided filter / dense CRF on probs with RGB guide) esp. PixelSeal flat holes. Kill: OOF.
5. Decoder-in-loop extremal-perturbation mask (Fong 1910.08485) -- low prior given failed gradient attribution.
6. Shift TTA (no flips: decoders not flip-equivariant), SWA/EMA.
