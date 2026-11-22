# Image-forensics localisers for generative edits when the whole image is VAE round-tripped ("fully regenerated", FR)

Notes written 21-22 Nov 2026. About 25 web calls. "Fetched" means I read the page this session; "from memory" means not re-verified this session and must be checked before relying on it.
Terminology: SP = spliced (generated region pasted back onto original pixels); FR = fully regenerated (whole latent decoded by the VAE, background not pixel-exact). Our task is the FR case.

## Ranked shortlist: what deserves a 15-minute kill test

### Takeaway
Off-the-shelf localisers are near-useless on FR images (F1 0.03-0.34), but the same architectures fine-tuned on FR data from the matching generator family reach F1 0.75-0.90 on FLUX. So the only pretrained weights worth a kill test are ones trained on FR/FLUX data (DinoLizer), and the highest-expected-value route is fine-tuning TruFor (or a SegFormer-style segmenter) on our own synthetic FR data, ideally made with a FLUX-family generator rather than SD2.

### Cited Findings
Ranking (1 = test first). Numbers are the papers' own; none were reproduced here.

1. **DinoLizer (DINOv2 ViT-B/14 + registers, LoRA r=64, linear head, 3 classes real / VAE-only / inpainted)** — the only found model with downloadable weights that was trained on FR inpainting including FLUX.
   - Trained on B-Free (51k real, 309k synthetic) plus ~118k TGIF images; TGIF training data includes Flux.1 dev, schnell and Fill. Training crops 504x504, patch 14; inference is a sliding window with stride 128, small images upscaled to at least 1016x1016 — [DinoLizer arXiv 2511.20722v2, 2025](https://arxiv.org/html/2511.20722v2)
   - Reported IoU / F1: TGIF 0.60 / 0.68; SAGI-FR 0.29 / 0.40; FOSSIL-FR 0.45 / 0.53; SAGI-SP 0.56 / 0.64; CocoGlide 0.74 / 0.82; average over 9 sets 0.55 / 0.64, vs TruFor (original weights) average 0.34 / 0.41. No per-generator FLUX breakdown is given — [DinoLizer arXiv 2511.20722v2](https://arxiv.org/html/2511.20722v2)
   - 3-class (real / VAE / inpainted) vs 2-class: IoU/F1 0.55/0.64 vs 0.51/0.60, but the two runs also differ in loss, so the comparison is not fully controlled — [DinoLizer arXiv 2511.20722v2](https://arxiv.org/html/2511.20722v2)
   - Weights: README links https://nextcloud.univ-lille.fr/index.php/s/HS6CLT73DZpZEyq; inference `python test.py --stride 128 --folder image_examples/`; no licence file listed — [github.com/anonyme610/dinolizer](https://github.com/anonyme610/dinolizer)
   - Robust to double JPEG down to QF 40, sensitive to Gaussian noise — [DinoLizer arXiv 2511.20722v2](https://arxiv.org/html/2511.20722v2)

2. **TruFor fine-tuned on our own FR data (not a download; a training recipe with strong published support).**
   - TGIF2: TruFor fine-tuned on all FR subsets reaches F1 (semantic / random masks) 0.87/0.94 on FLUX schnell, 0.82/0.88 on FLUX dev, 0.76/0.85 on FLUX Fill, 0.68/0.83 SDXL, 0.49/0.39 SD2; average 0.75. MMFusion fine-tuned: average 0.74. Original TruFor on the same FR sets: 0.33/0.09, 0.25/0.06, 0.16/0.08 (schnell, dev, Fill) — [TGIF2 arXiv 2603.28613, Table 4, 2026](https://arxiv.org/html/2603.28613v1)
   - Generator match matters a lot: TruFor fine-tuned on FLUX only gets 0.90/0.90, 0.86/0.81, 0.78/0.79 on schnell/dev/Fill; fine-tuned on SD2 only gets 0.40/0.16, 0.31/0.08, 0.21/0.10 on the same FLUX sets; fine-tuned on SDXL only gets 0.19/0.02, 0.08/0.01, 0.06/0.02 — [TGIF2 Table 5](https://arxiv.org/html/2603.28613v1). The fetch tool flagged this table as partly garbled; verify cells in the PDF before quoting.
   - SAGI-D: retraining raises FR mean IoU from 19.6 to 72.4 (TruFor), 16.9 to 50.3 (MMFusion), 5.9 to 36.2 (CAT-Net), 9.2 to 19.6 (PSCC-Net) — [SAGI arXiv 2502.06593v3, Table 5, ICCV 2025](https://arxiv.org/html/2502.06593v3)
   - TGIF2 fine-tune recipe: two stages of 100 epochs (localisation, then detection head), original hyperparameters, best validation loss; no fine-tuned weights URL published — [TGIF2](https://arxiv.org/html/2603.28613v1)

3. **TruFor original weights (already run by the team)** — expect weak results; it is still the best un-fine-tuned classical localiser on FR.
   - FR F1 with original weights: SD2 0.19, SDXL 0.18 — [TGIF arXiv 2407.11566v2, Table I, 2024](https://arxiv.org/html/2407.11566v2); FLUX schnell 0.33, dev 0.25, Fill 0.16 on semantic masks, 0.06-0.09 on random masks — [TGIF2 Table 4](https://arxiv.org/html/2603.28613v1)
   - SAGI-D FR mean IoU with original weights: TruFor 19.6, MMFusion 16.9, PSCC-Net 9.2, CAT-Net 5.9 — [SAGI Table 5](https://arxiv.org/html/2502.06593v3)

4. **DeCLIP (frozen CLIP ViT-L/14 features, optionally + RN50, conv decoder)** — weights exist, but trained on face inpainting (Dolos) and COCO-SD, not FLUX.
   - Claims localisation "on the challenging case of latent diffusion models where the entire image is affected by the fingerprint of the generator" — [DeCLIP arXiv 2409.08849, WACV 2025](https://arxiv.org/html/2409.08849v2)
   - Checkpoints (LDM, LaMa, Pluralistic, RePaint, COCO-SD), e.g. https://storage.cloud.google.com/bitdefender_ml_artifacts/declip/backbone_VIT%2BRN50/ViT_layer20%2BRN50_layer3_conv-20_ldm.pth and https://storage.cloud.google.com/bitdefender_ml_artifacts/declip/backbone_VIT/ViT_layer20_conv-20_cocosd.pth; code licence CC BY-NC-SA 4.0; eval via `python validate.py --arch=CLIP:ViT-L/14 --ckpt=... --fully_supervised` — [github.com/bit-ml/DeCLIP](https://github.com/bit-ml/DeCLIP)

5. **SAFIRE (SAM-based, point-prompted)** — weights exist via the repo, but there is no FR evidence in its paper, and DinoLizer's benchmark ranks it below TruFor.
   - CocoGlide F1 fixed/best 63.5/76.9 vs TruFor 52.2/72.0; average over 5 classical sets 62.6/73.0 vs TruFor 56.3/68.7 — [SAFIRE arXiv 2412.08197, AAAI 2025](https://arxiv.org/html/2412.08197v1)
   - In DinoLizer's benchmark of generative inpainting sets, SAFIRE, SparseViT, ManTra-Net, CAT-Net v2 and DINOv3-IML all average below TruFor's IoU 0.34 — [DinoLizer](https://arxiv.org/html/2511.20722v2)
   - Code: [github.com/mjkwon2021/SAFIRE](https://github.com/mjkwon2021/SAFIRE) (weights link not confirmed this session)

6. **DiffusionPrint (learned noise fingerprint replacing Noiseprint++ inside TruFor/MMFusion)** — most on-target method, but NOT usable today: no weights.
   - README: "Pretrained DiffusionPrint checkpoints will be released shortly"; only the patch dataset is public (huggingface.co/datasets/giakoupg/diffusionprint_dataset) — [github.com/mever-team/diffusionprint](https://github.com/mever-team/diffusionprint)
   - TGIF FR F1 (semantic / random), TruFor with Noiseprint++ -> with DiffusionPrint, both fine-tuned on TGIF2 FR: SD2.1 50->53 / 23->25; SDXL 61->67 / 62->73; Flux schnell 80->89 / 62->85; Flux dev 69->83 / 46->74; Flux Fill 65->75 / 59->74 — [DiffusionPrint arXiv 2604.12443, 2026](https://arxiv.org/html/2604.12443v1)

Do not bother (evidence of failure on FR): CAT-Net (FR F1 0.03-0.05, because it relies on JPEG/DCT traces that regeneration removes; it is the best on SP at 0.86-0.93), PSCC-Net (0.02-0.05), ManTra-Net (0.02-0.05), SPAN (0.00), MVSS-Net++ (0.06-0.15), ImageForensicsOSN (0.09-0.30) — [TGIF2 Tables 3-4](https://arxiv.org/html/2603.28613v1), [TGIF Table I](https://arxiv.org/html/2407.11566v2)

### Inferences
- Our DINOv2 linear probe failing (AUC 0.45) does not contradict DinoLizer: DinoLizer starts from a DINOv2 already fine-tuned for synthetic-image detection on B-Free and adds LoRA, i.e. frozen generic DINOv2 features are not the claimed source of signal. Inference from the method description.
- DinoLizer's "VAE" class is useful to us in both directions: for hole-type images, inpainted-class probability is the mask; for the inverse case (original rectangle inside a generated image), the complement of the inpainted class within the image. If every pixel in our data is VAE-decoded, the "real" class should be near-empty; if it is not, that itself tells us compositing happened. Speculation, untested.
- Since our images are 512 px and DinoLizer upsamples to 1016 with stride-128 windows, expect roughly 25 windows per image; cheap enough for 510 images in a 15-minute test. Inference.
- The TGIF2 cross-generator table is a warning for our SD2-inpaint + FLUX-VAE synthetic generator: SD2-only fine-tuning transferred poorly to FLUX (F1 0.21-0.40 semantic, 0.08-0.16 random). Our FLUX-VAE round trip removes the VAE mismatch but not the denoiser mismatch; whether that is enough is untested. If FLUX.1 Fill/schnell weights can be downloaded, generating synthetic data with them is the better-supported choice.

### Gaps
- Not found: any released TruFor/MMFusion checkpoint fine-tuned on TGIF2 FR or SAGI-D. The SAGI repo page (Apache-2.0) shows code and a Kaggle dataset but no weights link — [github.com/mever-team/SAGI](https://github.com/mever-team/SAGI). The TGIF2 paper gives only the dataset repo — [github.com/IDLabMedia/tgif-dataset](https://github.com/IDLabMedia/tgif-dataset).
- DINO-FIL (DINOv3 ViT-H+/16 multi-layer fusion, trained on B-Free, tested on Beyond the Brush, CocoGlide, TGIF, SAGI-D; Oct 2026): abstract only, no numbers, no code or weights link — [arXiv 2610.21780](https://arxiv.org/abs/2610.21780).
- Not researched for lack of time (no evidence either way gathered this session): GIM, COCOInpaint, BR-Gen, OpenSDI, FakeShield, SIDA, LEGION, DiQuID, HiFi-Net, IML-ViT, Mesorch, SparseViT weights, IMDL-BenCo model zoo, End4 ([arXiv 2509.13214](https://arxiv.org/html/2509.13214v1), seen in search only), EDGER ([arXiv 2605.12002](https://arxiv.org/pdf/2605.12002), seen in search only).
- From memory, unverified this session: TruFor weights at https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip; MMFusion at github.com/idt-iti/MMFusion-IML; CAT-Net at github.com/mjkwon2021/CAT-Net; IMDL-BenCo at github.com/scu-zjz/IMDLBenCo.
- DinoLizer licence is unstated, and the weights host is a university Nextcloud share; download was not tested.

## TGIF / TGIF2 exact findings: spliced vs fully regenerated

### Takeaway
Every pretrained IFL method collapses on FR: the best original-weights F1 on FR is about 0.2-0.34, against 0.75-0.89 on SP for the same content. Fine-tuning on FR data recovers F1 of about 0.75 on average and up to 0.9 on FLUX, with a measurable object-shape bias.

### Cited Findings
- TGIF (2024), F1, columns SD2-SP / Firefly-SP / SD2-FR / SDXL-FR: CAT-Net 0.87 / 0.85 / 0.04 / 0.03; TruFor 0.83 / 0.79 / 0.19 / 0.18; MMFusion 0.75 / 0.74 / 0.15 / 0.18; ImageForensicsOSN 0.23 / 0.36 / 0.20 / 0.18; PSCC-Net 0.15 / 0.38 / 0.05 / 0.05; ManTra-Net 0.15 / 0.56 / 0.03 / 0.05; MVSS-Net++ 0.07 / 0.08 / 0.06 / 0.09; SPAN 0.00 throughout. Best on FR: ImageForensicsOSN 0.20 (SD2), TruFor/MMFusion/OSN 0.18 (SDXL) — [TGIF arXiv 2407.11566v2, Table I](https://arxiv.org/html/2407.11566v2)
- Stated reason for the FR failure: regeneration removes compression artefacts and camera noise that IFL methods rely on — [TGIF](https://arxiv.org/html/2407.11566v2)
- TGIF construction: COCO val2017 images, one object mask (segmentation mask or bounding box) per image, the object replaced by a generated one of the same class; SD2 at 512x512 saved both as SP (blended with gradient border) and FR; SDXL FR only; Firefly SP only; about 75k forged images — [TGIF](https://arxiv.org/html/2407.11566v2)
- TGIF2 (arXiv 30 Mar 2026, accepted at a journal on information security) adds FLUX.1 schnell, dev and Fill dev, plus random non-semantic masks (rectangles, at least 64 px, at most 60% of width/height, empty prompt) — [TGIF2](https://arxiv.org/html/2603.28613v1)
- TGIF2 FR F1, original weights, semantic / random: TruFor schnell 0.33/0.09, dev 0.25/0.06, Fill 0.16/0.08; MMFusion 0.34/0.08, 0.20/0.04, 0.12/0.06; OSN 0.30/0.08, 0.23/0.07, 0.09/0.05; CAT-Net 0.05/0.01, 0.04/0.01, 0.03/0.02 — [TGIF2 Table 4](https://arxiv.org/html/2603.28613v1)
- TGIF2 SP F1 for comparison: CAT-Net average 0.89, TruFor 0.75, MMFusion 0.64 — [TGIF2 Table 3](https://arxiv.org/html/2603.28613v1)
- Object bias: fine-tuning on semantic masks only lowers random-mask performance (TruFor SD2 0.50 semantic vs 0.23 random); adding random masks to fine-tuning fixes random without hurting semantic; an example shows fine-tuned TruFor finding the inpainted tennis racket but missing the random box — [TGIF2](https://arxiv.org/html/2603.28613v1)
- FLUX Fill dev is the hardest FLUX variant after fine-tuning (0.76/0.85 vs 0.87/0.94 for schnell) — [TGIF2 Table 4](https://arxiv.org/html/2603.28613v1)
- Generative super-resolution (Real-ESRGAN x2/x4 then downsample) destroys SP localisation: CAT-Net 0.87 -> 0.06, TruFor 0.78 -> 0.21 — [TGIF2 Table 9](https://arxiv.org/html/2603.28613v1)
- Image-level SID on FR: B-Free AUC 1.00 on SD2, 0.78-0.88 on FLUX subsets — [TGIF2 Table 8](https://arxiv.org/html/2603.28613v1)

### Inferences
- Original-weights detectors score much better on semantic than random masks on FR (TruFor 0.22 vs 0.08 average), i.e. part of their residual "success" is object saliency, not forensics. For our object-shaped holes this bias helps; for our non-object original rectangles it hurts. Train with both shapes.
- Our "original rectangle inside generated image" case corresponds to TGIF2's random rectangles with inverted labels; fine-tuned TruFor reached F1 0.85-0.94 on FLUX random rectangles, so rectangles are learnable when the training generator matches.

### Gaps
- FR IoU tables (TGIF2 Tables 12-13) were beyond the fetched portion; only F1 is reported above.
- TGIF2 does not describe latent or mask handling for FLUX schnell/dev (non-inpainting models "used without adaptation"); not found.
- Minimum mask size in TGIF and results stratified by mask area: not extracted. Our median 6% area may be smaller than TGIF objects.

## Is a diffusion-generated region distinguishable from a VAE-only reconstructed region?

### Takeaway
Yes, by learned detectors with supervision: three independent 2025-2026 works build on exactly this distinction and reach F1 0.75-0.9 on FLUX FR data. There is no evidence that a training-free cue (reconstruction error, spectra) separates the two, and one study attributes 97% of the background spectral shift to the VAE alone, so naive spectral or reconstruction-error maps will mostly see "VAE everywhere".

### Cited Findings
- VAE round trip alone triggers detectors: a real photo passed through the autoencoder with no synthesis is flagged fake 91% of the time by Corvi2023 (standard inpainting 0.89; original background restored 0.11). A variance decomposition attributes 97.3% of the background spectral shift to the autoencoder round trip and 1.6% to the denoiser — [INP-X arXiv 2602.00192, 2026](https://arxiv.org/html/2602.00192)
- Some signal does live in the generated content itself: mask-only detection is about 0.75 and unchanged whether the background is VAE-reconstructed or restored (0.747 vs 0.753), while background-only detection drops 0.949 -> 0.561 when original pixels are restored — [INP-X](https://arxiv.org/html/2602.00192)
- Saliency-based localisation from fine-tuned classifiers stays under 0.5 mIoU (EfficientNet 0.466-0.486); CNN backbones localise better than ViTs — [INP-X](https://arxiv.org/html/2602.00192)
- DiffusionPrint's design: VAE reconstruction shifts real patches' high-frequency spectra toward generated content, so simple noise statistics fail; it trains a 17-layer DnCNN contrastively with VAE-reconstructed authentic patches as positives of real ones (invariance to the VAE) and real patches at inside-mask positions as hard negatives; trained on 64x64 patches from SD2.1, SDXL and Flux.1 — [DiffusionPrint](https://arxiv.org/html/2604.12443v1)
- Evidence the learned fingerprint beats camera-noise fingerprints on FR: TruFor F1 on Flux dev 69 -> 83 (semantic), 46 -> 74 (random) when Noiseprint++ is swapped for DiffusionPrint; a variant pretrained without Flux still beats Noiseprint++ on several Flux subsets (87 vs 80 on schnell semantic) — [DiffusionPrint](https://arxiv.org/html/2604.12443v1)
- DinoLizer treats auto-encoded regions as a third class; gain 0.51 -> 0.55 IoU over two classes (not fully controlled) — [DinoLizer](https://arxiv.org/html/2511.20722v2)
- AEROBLADE (CVPR 2024) reports that autoencoder reconstruction error "gives a good indication" of where the inpainted region is, but in the setting of real images with inpainted regions (unedited part not VAE-decoded), not FR — [AEROBLADE arXiv 2401.17879](https://arxiv.org/pdf/2401.17879) (via search summary, paper not re-read this session)
- DIRE and LaRE use diffusion reconstruction error for image-level detection — search summary only; primary papers not fetched.

### Inferences
- Second round-trip (idempotence) with the same FLUX VAE is unlikely to localise in our setting: both the inpainted region and the background are outputs of the same decoder, so both should re-encode with similarly low error. The 97.3% / 1.6% decomposition supports this for spectral statistics. A small residual difference could exist because denoised latents are "on the prior manifold" whereas encoded-real latents are not, but that is speculation with no published measurement found.
- The plausible physical cue is noise/texture content: an encoded real photo carries (attenuated) sensor noise and fine texture through the VAE, whereas denoised latents decode to cleaner, model-typical texture. DiffusionPrint's success with a noise-residual DnCNN and Noiseprint++'s partial success after fine-tuning are consistent with this. Inference, not stated as such by the papers.
- A cheap per-latent-cell experiment that is consistent with the above (speculative, 15 minutes): encode with the FLUX VAE, take the 16-channel 64x64 latent, and train a tiny 1x1/3x3 conv classifier on our 510 labelled images. If the latent itself carries the cue, this is the fastest detector to train; if AUC is near 0.5, drop it.
- Our images carry an invisible watermark in the original regions. If the whole image was VAE round-tripped after watermarking, the watermark is attenuated in the background but absent from generated regions; that is a task-specific cue outside the forensic literature reviewed here.

### Gaps
- Not found: any paper that uses the FLUX 16-channel VAE (or any VAE inversion) to produce a per-latent-cell (64x64) anomaly map for localisation. Targeted search returned only DiffusionPrint, DinoLizer and INP-X.
- Not found: any measurement of second-round-trip reconstruction error for generated vs VAE-reconstructed regions within the same image.
- DRCT, FIRE: not researched this session.
- INP-X benchmark uses Kandinsky 2.2, OpenJourney and SD 1.4; FLUX.1 appears only as an unseen-generator test, so the 97.3% figure is not FLUX-specific.

## Boundary / seam cues and what non-grid-aligned rectangles imply

### Takeaway
FLUX.1 Fill receives the mask at full pixel resolution (folded 8x8 into 64 channels), never composites original pixels or latents, and decodes the whole latent, so mask edges need not align to the 8 or 16 px grid and no hard seam is expected. Non-aligned rectangles are therefore consistent with a plain FLUX Fill call and do not by themselves indicate paste-then-img2img.

### Cited Findings
- diffusers `FluxFillPipeline`: mask reshaped as `mask.view(batch_size, height, self.vae_scale_factor, width, self.vae_scale_factor)` into 8*8 channels at latent resolution (not downsampled); masked image `init_image * (1 - mask_image)` is VAE-encoded; conditioning is concatenated to the transformer input (`hidden_states=torch.cat((latents, masked_image_latents), dim=2)`); the denoising loop has no mask-based blending; the full latent is decoded with `self.vae.decode(latents)`; height/width are floored to multiples of 16 — [diffusers pipeline_flux_fill.py](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/flux/pipeline_flux_fill.py)
- TGIF's SP variant blends the inpainted region back "with a gradient border", i.e. spliced pipelines soften the seam deliberately — [TGIF](https://arxiv.org/html/2407.11566v2)
- Soft alpha-blending the seam did not change detectability in the INP-X study (detection stays low when original background is restored) — [INP-X](https://arxiv.org/html/2602.00192)
- FLUX schnell/dev in TGIF2 are non-inpainting models "used without adaptation" for FR inpainting — [TGIF2](https://arxiv.org/html/2603.28613v1)

### Inferences
- Because the masked image is zeroed inside the mask before VAE encoding, the encoder sees a sharp black edge; latent cells straddling the mask border (a band of roughly 8-16 px, plus the 2x2 patchification, i.e. a 16 px token grid) are conditioned on mixed content. A border band where the output deviates from the true original more than deep background is plausible, but no paper measuring this was found.
- If the edits were instead made with a non-Fill FLUX model via latent blending (the usual diffusers inpaint approach, where the mask is interpolated to 64x64 latent resolution and original noised latents are re-injected each step), the effective mask would be quantised to 8 px cells and the generated/kept boundary would follow the latent grid even if the nominal rectangle does not. From memory of diffusers `FluxInpaintPipeline` / SD inpaint (legacy) behaviour; not re-verified this session.
- Diagnostic that follows: on training images, measure where |image - best-guess original| or the label boundary falls relative to the 8 px and 16 px grids. Grid-snapped ground truth suggests latent blending; pixel-accurate ground truth with a soft transition suggests a Fill-style channel-conditioned model. Our observation of non-aligned rectangles points to the latter, or to masks defined before a resize.

### Gaps
- Not found: published measurements of seam artefacts at the mask border in latent-diffusion FR inpainting, or any localiser built on an 8 px-grid seam cue.
- Not found: literature on inferring the outpainting procedure from mask alignment.

## SAM / SAM2 fusion to snap forensic maps to object boundaries

### Takeaway
There is published evidence that SAM-based forensic models beat TruFor on classical and GLIDE-inpainting benchmarks, but no found ablation isolating "heat map + SAM snapping" as post-processing on FR inpainting; on generative-inpainting benchmarks SAFIRE scores below TruFor. Treat SAM snapping as a plausible but unproven post-process to validate on our 510 images.

### Cited Findings
- SAFIRE ablation (average F1 fixed / best over 5 datasets): full 62.6 / 73.0; prompting replaced by plain binary segmentation 28.0 / 37.4; region-to-region pretraining replaced by SAM pretraining 48.6 / 61.0; vanilla SAM baseline 22.3 / 34.1 — [SAFIRE Table 2](https://arxiv.org/html/2412.08197v1)
- SAFIRE on CocoGlide (GLIDE inpainting, spliced-style): 63.5 / 76.9 vs TruFor 52.2 / 72.0; on SafireMS-Expert two-source partition mIoU 90.3 vs TruFor 82.5 — [SAFIRE Tables 1, 3](https://arxiv.org/html/2412.08197v1)
- SAFIRE averages below TruFor (IoU 0.34) on DinoLizer's nine generative-inpainting sets — [DinoLizer](https://arxiv.org/html/2511.20722v2)
- Detective SAM (ICLR 2026 submission) turns a forensic perturbation signal into a heat-map prompt for a frozen SAM2 with adapters; reports average 34.68 IoU / 42.03 F1 out of distribution, a 38.94% relative IoU gain over the best baseline — [OpenReview PDF](https://openreview.net/pdf/e5866e1f442d0de185ec477b4aec34521a6c354c.pdf) (search summary only, not read)
- SARIF (2026) feeds the previous mask back to SAM as a prompt for iterative boundary refinement — [arXiv 2606.21108](https://arxiv.org/pdf/2606.21108) (search summary only)
- SAMRefiner is a generic SAM-based mask-refinement post-processor, not forensic — [arXiv 2502.06756](https://arxiv.org/pdf/2502.06756) (search summary only)
- Object bias cuts both ways: detectors fine-tuned on object masks miss non-object rectangles — [TGIF2](https://arxiv.org/html/2603.28613v1)

### Inferences
- For our object-shaped holes (object removal/addition), choosing among SAM/SAM2 automatic masks the one(s) maximising mean forensic score inside minus outside is a cheap post-process that should raise IoU when the heat map is roughly right but blurry (DinoLizer's native resolution is 14 px patches). Not supported by a direct published ablation.
- For removed objects the generated region has the shape of the removed object, which is no longer visible; SAM on the edited image will not find that boundary. Snapping should help object addition/replacement and transplantation, and may hurt removal. For rectangles, fit an axis-aligned box to the thresholded map instead.

### Gaps
- Not found: numbers for SAM snapping as pure post-processing on FR inpainting data.
- SAFIRE weights URL and licence not confirmed; the paper states release for free research use without naming a licence — [SAFIRE](https://arxiv.org/html/2412.08197v1)

## Training a small-region segmenter on few real + many synthetic images

### Takeaway
The forensic papers reviewed give four concrete, evidenced practices: match the generator family, mix object-shaped and random-rectangle masks, treat VAE-only background as its own class or as an explicit invariance, and cap class weights. Loss choices for tiny regions rest on general segmentation literature, not on FR-specific ablations.

### Cited Findings
- Match the generator: TruFor fine-tuned on FLUX-only FR data gets F1 0.78-0.90 on FLUX; fine-tuned on SD2-only gets 0.21-0.40 (semantic) and 0.08-0.16 (random) on FLUX — [TGIF2 Table 5](https://arxiv.org/html/2603.28613v1)
- Mix mask shapes: semantic-only fine-tuning drops random-mask F1 (SD2 0.50 vs 0.23); semantic + random gives the best of both (average 0.72 / 0.78) — [TGIF2 Table 5](https://arxiv.org/html/2603.28613v1)
- Balance generators when pooling (TGIF2 samples equal numbers from SD2, SDXL, FLUX) — [TGIF2](https://arxiv.org/html/2603.28613v1)
- Explicit VAE-only class with weighted cross-entropy, class weights capped at 50, per-patch labels, LoRA rank 64 on a detection-pretrained DINOv2 (2.4M trainable parameters), 504 px crops, sliding-window inference with upscaling of small images to at least 1016 px — [DinoLizer](https://arxiv.org/html/2511.20722v2)
- Hard negatives: real patches sampled at inside-mask positions; positives: VAE-reconstructed versions of authentic patches; frozen fingerprint backbone, only fusion and decoder heads fine-tuned, 512x512 random crops — [DiffusionPrint](https://arxiv.org/html/2604.12443v1)
- A lightweight baseline (SegFormer MiT-B2 RGB stream + fingerprint stream, about 36M parameters) reaches F1 80 / 73 on Flux dev FR with DiffusionPrint features, 65 / 62 with Noiseprint++ — [DiffusionPrint](https://arxiv.org/html/2604.12443v1)
- Retraining on a large matched FR set improves out-of-domain IoU too (TruFor 19.1 -> 65.5 OOD on SAGI-D) — [SAGI Table 5](https://arxiv.org/html/2502.06593v3)
- CNN backbones localise better than ViTs in saliency-based localisation — [INP-X](https://arxiv.org/html/2602.00192)

### Inferences
- Synthetic-data recipe implied by the above: for each real photo produce (a) a VAE-only round trip (all-negative image, also the source of hard negatives), (b) FR inpaints with object masks, (c) FR inpaints with random rectangles not aligned to the 8 px grid, (d) inverted cases (generate most of the image, keep a rectangle/object). Use the FLUX VAE for every image and a FLUX-family denoiser if obtainable.
- General segmentation practice for tiny regions, from memory and not FR-specific: BCE or focal loss ([Lin et al. 2017, arXiv 1708.02002](https://arxiv.org/abs/1708.02002)) combined with a per-image Dice ([Milletari et al. 2016, arXiv 1606.04797](https://arxiv.org/abs/1606.04797)) or Lovasz hinge IoU surrogate ([Berman et al. 2018, arXiv 1705.08790](https://arxiv.org/abs/1705.08790)); Tversky loss to trade recall for precision on small lesions ([Salehi et al. 2017, arXiv 1706.05721](https://arxiv.org/abs/1706.05721)); copy-paste augmentation ([Ghiasi et al. 2021, arXiv 2012.07177](https://arxiv.org/abs/2012.07177)). These links were not fetched this session.
- Caution on copy-paste: pasting pixels in image space creates a splice seam and pixel-exact content that our real data does not have; if used, paste before the VAE round trip (paste generated content into a real image, then encode/decode the whole image) so the composite has the same global VAE trace. Inference from the SP-vs-FR results.
- Compute the Dice/Lovasz term per image, not per batch, if the metric is per-image IoU, and include all-negative images with a BCE-only term; keep full 512 px resolution (do not downsample: a quarter of our masks are under 2.3% of the image, about 77x77 px, i.e. about 5x5 DINOv2 patches or 10x10 latent cells). Arithmetic and inference.
- With only 510 real labelled images, use them mainly for validation, threshold selection and a final low-learning-rate fine-tune; the SD2-only -> FLUX transfer numbers suggest real-distribution images are worth far more per sample than mismatched synthetic ones.

### Gaps
- Not found: FR-specific ablations of loss functions, input resolution or copy-paste augmentation for small masks.
- Not found: any study of few-shot adaptation (hundreds of target-generator images) of an FR localiser; the TGIF2 and SAGI results use tens of thousands of images.
