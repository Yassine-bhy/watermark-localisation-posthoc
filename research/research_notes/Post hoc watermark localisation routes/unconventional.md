# Unconventional / cross-field routes for post hoc watermark localisation

Reading guide. "Cited Findings" are only what a source says. Everything that maps a source onto our problem is under "Inferences" and is NOT evidence that it works here. Sources marked (memory) were not re-fetched in this session: the URL and the gist are from prior knowledge and should be treated as less certain than the fetched ones. Time-box was about 20 minutes, 17 tool calls.

Ranked shortlist (all inference, ordered by expected value per hour):
1. Pre-pooling "known-message CAM": apply the extractor's last layer per spatial location and project on the known message (Q5). Minutes to implement, no training.
2. Max-sum rectangle (exact, integral image / 2D Kadane) on whatever score map exists, to turn blurry maps into sharp rectangles in regime 2 (Q4), followed by CMA-ES refinement of the 4 box parameters on the decoder's known-message margin (Q3).
3. "Null-balanced" counterfactual: blend the image with a copy re-embedded with the complementary message so the decoder sits at its decision boundary, then do gradients / RISE regression there (Q1, Q2).
4. Per-image simulated supervision: punch synthetic holes / paste synthetic rectangles into the test image itself and fit a tiny head (Q5).
5. Decision rule: never output an empty minority class; threshold below 0.5 (Q6).

## Mask optimisation through the differentiable extractor (meaningful / extremal perturbations, RISE, fills)

### Takeaway
The literature supports optimised masks with an explicit area constraint and enforced smoothness over raw gradients, but only for classifiers that are not saturated; our extractors are heavily redundant, so the fill must be an active counter-signal (a copy re-embedded with a different message), not blur. No source was found that applies mask optimisation to a frozen watermark extractor.

### Cited Findings
- Extremal perturbations define the mask of a FIXED area that has the largest effect on the output; the area is enforced by a ranking-based loss acting like a hard constraint, replacing the soft area penalty of Fong and Vedaldi 2017 that needed several balanced energy terms — [Fong, Patrick, Vedaldi, ICCV 2019](https://arxiv.org/abs/1910.08485)
- The same paper draws masks from a parametric family with guaranteed minimum smoothness (smooth-max convolution), computes masks at areas from 5% to 80% of the image, and optimises with SGD starting from all ones — [Fong, Patrick, Vedaldi, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fong_Understanding_Deep_Networks_via_Extremal_Perturbations_and_Smooth_Masks_ICCV_2019_paper.pdf)
- RISE estimates saliency as a weighted sum of random masks, where masks are sampled on a low-resolution grid with keep-probability p, upsampled and randomly shifted — [Petsiuk, Das, Saenko, 2018](https://arxiv.org/abs/1806.07421)
- A practical setting documented for the detector variant D-RISE: 16,384 samples with an 8x8 mask grid and mask probability 0.85 when more precision is wanted — [MathWorks D-RISE documentation, accessed 2026](https://www.mathworks.com/help/deeplearning/ref/drise.html)
- Original meaningful-perturbation formulation (learn a mask with area + total-variation penalties, deletion and preservation games, blur / noise / constant fills) — [Fong and Vedaldi, ICCV 2017](https://arxiv.org/abs/1704.03296) (memory)
- MaskMark (NeurIPS 2025) trains its decoder with a mask applied to the image BEFORE extraction so that the decoder learns local extraction; it is a trained scheme, not a post hoc method on an existing decoder — [Hu et al., 2025](https://arxiv.org/abs/2504.12739)

### Inferences
- Why occlusion and gradients gave nothing (inference): in regime 1 the remaining 95% of the image still carries the message, so the margin is saturated and deleting any small region changes nothing; in regime 2 the pattern only decodes when the rectangle is resampled to model resolution, so in-place masking never produces a decodable input. Mask optimisation in the image frame therefore only makes sense for regime 1; regime 2 needs box parameters (Q3).
- Fill choice (inference). Blur / inpaint / VAE-regenerated fills only remove evidence, which a saturated decoder ignores. Use fill = Emb(y, not-m) (complementary message) so replaced pixels vote AGAINST m. Then for z = M*y + (1-M)*y_anti, the margin s(z) = sum_i (2m_i-1)*logit_i(z) behaves roughly like "watermarked area kept minus area replaced". Watermarked pixels are worth about +1 when kept and clean pixels about 0, while both are worth about -1 when replaced. This two-level contrast is the localisation signal.
- Operating point (inference, the key trick). Start M at a level where s(z) is near 0 (uniform M around 0.5, or a random half/half checkerboard at 16 px). There the decoder is in its linear regime, and ds/dM (one backward pass, average over 16 to 64 random checkerboards / jitters, SmoothGrad-style) is a per-pixel evidence map. This is a bridge / null measurement; expected cost 64 forward-backward passes per image.
- Extremal variant for holes (inference): fix area a in {1, 2, 4, 8}% from the area prior, optimise M (128x128, generated from a 32x32 or 64x64 latent with Gaussian upsampling) to MAXIMISE s when the complement of M is kept and M is replaced by y_anti... the hole is the region whose replacement costs the least margin. Equivalent and cheaper: hole = lowest-a-quantile of the gradient map above, then TV-smooth.
- RISE sample count (inference from a variance argument, not from a source): with K grid cells each toggled independently, the output variance is dominated by the K-1 other cells, so the per-cell estimate needs N on the order of K times (contrast ratio)^-2 samples. For a 16x16 grid (cell = 3% of area... each cell 0.4%) and a 2x contrast this means several thousand to about 16k forward passes per image, consistent with the D-RISE setting. Fit by ridge regression of s on the mask indicators rather than the RISE weighted mean; it removes the between-cell variance. A 1% region needs at least a 16x16 grid; an 8x8 grid cannot resolve it.
- Failure modes: (a) if re-embedding does not overwrite (see Q2: PixelSeal evidence is unverified), y_anti is a weak counter-signal and the balance point does not exist; check in 5 minutes by decoding Emb(y, not-m). (b) The extractor resizes to 256/245/400, so one mask pixel at 128 is 2 to 3 decoder pixels; the map cannot be sharper than the extractor's receptive field. (c) Decoders may respond to blend seams; use soft masks (Gaussian sigma about 4 px at 512).

### Gaps
- No source found that measures localisation sharpness of extremal perturbations versus gradients specifically for globally pooled BIT decoders; all evidence is for ImageNet classifiers.
- No source found for RISE sample complexity as a function of object area; the count above is my own estimate.

## Counterfactual re-embedding tests and watermark coexistence / overwriting

### Takeaway
There is direct evidence that re-embedding with the SAME method and a different message overwrites the first message (including TrustMark), while different methods mostly coexist; nothing was found on how decoders react LOCALLY, so local use is our own extrapolation.

### Cited Findings
- Across eight methods (DwtDct, DwtDctSvd, RivaGAN, SSL, HiDDeN, RoSteALS, TrustMark B and Q), embedding a second message with the same method makes the first unrecoverable: "decoding the first secret yields 0% accuracy across the diagonal"; TrustMark Q and B also overwrite each other — [Petrov et al., On the Coexistence and Ensembling of Watermarks, 2025](https://arxiv.org/html/2501.17356v1)
- Different methods usually coexist with a few points of accuracy lost; the worst case reported is RoSteALS cutting the first watermark by up to 76% after TrustMark Q. StegaStamp and VideoSeal are not covered. The mechanism is not established; the authors suggest "quasi-orthogonal subspaces" and call it an open question — [same paper, 2025](https://arxiv.org/html/2501.17356v1)
- "Simply re-watermarking an already watermarked image reliably suppresses the original signal", with bit accuracy reduced by at least 25% and up to 48% across 96 dataset / victim / attacker combinations (abstract only; methods not named there) — [Bulychev, Marchant, Rubinstein, Watermarks Attack Watermarks, May 2026](https://arxiv.org/abs/2605.16796)
- FiLM-CAM (Kulthe, Agarwal, Asnani, Gilbert, Collomosse; Oct to Nov 2026) designs an encoder to "estimate the watermark residual invariant to the presence of pre-existing watermarks", enabling removal by re-encoding and subtracting the residual. A search snippet attributed to this paper the statement that a second embedding drops first-watermark accuracy from 0.99 to 0.50 for TrustMark and PixelSeal; I could NOT verify this in the abstract (PDF too large to fetch) — [FiLM-CAM, 2026](https://arxiv.org/abs/2610.30347)
- A WACV 2026 post hoc block-level scheme (MELB) localises its OWN watermark with a per-block hypothesis test (96x96 blocks, one-sided test at 5%); it reports no IoU, only that 92 to 95% of watermarked area is flagged versus 17 to 19% on originals. It does not localise third-party watermarks — [Bulychev et al., Where is the Watermark?, 2025/2026](https://arxiv.org/html/2512.14994v1)

### Inferences
- Same-method overwriting is the evidence base for using y_anti = Emb(y, not-m) as the counter-signal fill in Q1. The FiLM-CAM statement (encoder residual must be explicitly trained to be invariant to pre-existing watermarks) hints that for ordinary encoders the residual DOES depend on what is already there; however our own measurement (local correlation with the re-embedded pattern gave nothing) says this dependence is not visible in pixel space after the VAE round trip.
- Test in decoder space rather than pixel space (inference): compute D(p) = change in known-message margin when a patch around p is replaced by the SAME-message re-embedding Emb(y, m) versus by Emb(y, not-m). In a region already carrying m, same-message replacement changes little and opposite-message replacement flips it; in a clean region both are "new information". Evaluate at the balanced operating point of Q1, otherwise saturation hides it. Cost: 2 embeds + batched decoding of patch composites (for a 16x16 grid: 512 forwards).
- Global regime detector (inference): decode y and Emb(y, m') for a random m'; the margin of m in y scaled against a fully watermarked simulation of the same image estimates the watermarked AREA FRACTION, which gives the area constraint for extremal masks and the per-image threshold in Q6.
- Failure mode: StegaStamp and PixelSeal overwriting behaviour is not covered by a verified source; measure it before building on it.

### Gaps
- No literature found on local (spatially resolved) decoder response under partial overwriting for VideoSeal / PixelSeal / WAM / TrustMark.
- Same-method overwrite numbers for StegaStamp and PixelSeal: not found in a verified source.

## Efficient search for a rectangle with a needle-like objective

### Takeaway
Branch-and-bound (ESS) needs a bound that a neural decoder score does not provide, so it applies to additive score maps only; for the decoder score the sample-efficient recipe is proposal from a score map, then a derivative-free 4-parameter optimiser (CMA-ES) on a continuous margin, with a statistical threshold that accounts for the number of boxes tried.

### Cited Findings
- Efficient Subwindow Search finds the globally maximal rectangle by branch and bound over sets of boxes, "typically in linear or even sublinear time" versus quadratic for sliding windows, provided an upper bound of the quality over a set of boxes is available (given for additive per-pixel / per-feature scores) — [Lampert, Blaschko, Hofmann, CVPR 2008 / TPAMI 2009](https://pub.ista.ac.at/~chl/papers/lampert-pami2009.pdf)
- Spatial transformer networks give a differentiable crop / affine resampling so that box parameters receive gradients — [Jaderberg et al., 2015](https://arxiv.org/abs/1506.02025) (memory)
- CMA-ES is a derivative-free optimiser designed for low-dimensional, ill-conditioned, non-convex continuous problems, with default population 4 + 3 ln(n), i.e. about 8 for n = 4 — [Hansen, The CMA Evolution Strategy: A Tutorial, 2016](https://arxiv.org/abs/1604.00772) (memory)
- Classical resynchronisation: Kutter builds the watermark from shifted copies so autocorrelation peaks reveal rotation and scale; Pereira and Pun (IEEE TIP 2000) recover general affine transforms from a template of spectral peaks via polar maps; log-polar mapping converts rotation and scale into shifts found by a matched filter — [Deguillaume, Voloshynovskiy, Pun (survey text), Univ. Geneva](https://access.archive-ouverte.unige.ch/access/metadata/19cbda36-b0dd-450f-b003-bc5a75a85f30/download); [Digimarc patent US7152021](https://patents.justia.com/patent/7152021)

### Inferences
- Size of the blind search (my arithmetic). A peak 4 px wide in translation and 5% in scale, with aspect up to 16:1, means roughly (512/4)^2 positions times (ln(16*8)/0.05)^2 scale pairs, on the order of 10^7 to 10^8 boxes: blind grid search is infeasible, and Gaussian smoothing of the objective (averaging the score over jittered boxes, ES-style) dilutes a needle by the volume ratio, so it does not create a usable gradient far from the peak. Gradient refinement through a differentiable crop is only useful INSIDE the basin (last 1 to 2 px), not for finding it.
- Recommended pipeline (inference): (1) proposal box from the max-sum rectangle on the segmentation map (Q4); (2) CMA-ES over (cx, cy, log w, log h) with sigma0 equal to the proposal uncertainty (about 10 to 15 px, 10% scale), population 32 to 64 evaluated as ONE decoder batch via roi_align, 30 to 50 generations, i.e. 1 to 3k decodes per image; restart from the top 3 proposals; (3) objective = continuous margin sum_i (2m_i-1)*logit_i on the resampled crop, not bit accuracy, because it has no plateau at chance. If the prior box is within +-15 px per edge, the basin is about (4/30)^4 = 3e-4 of the search volume, so even pure random search hits it in about 3k samples; CMA-ES should need fewer once any sample lands on the shoulder.
- Use the asymmetric basin (inference): since boxes slightly LARGER than the truth keep the score, bias the search outward: sample scale on a geometric ladder with ratio 1.05 from large to small, and after a hit run per-edge bisection (shrink one edge until the margin drops) to get each edge to 1 to 2 px: about 4 x 6 decodes.
- Significance threshold for "we found it" (my arithmetic): under the null, matches are Binomial(n, 0.5). With N boxes tried, the maximum null score is about n/2 + sqrt(n/4)*sqrt(2 ln N). For n = 100 bits and N = 3,000 this is about 70 bits; for N = 5e7 about 80 bits. For n = 256 and N = 3,000 about 160 bits (62.5%). Accept a box only above that; otherwise fall back to the scan rectangle.
- TrustMark has a wide basin, so a coarse grid (stride about 1/4 of box size, 1.25x scale steps) plus bisection suffices; spend the search budget on StegaStamp and PixelSeal.
- Spectral scale estimation (speculation, 10-minute check): average |FFT(Emb(x,m) - x)| over 100 images at model resolution. If the residual has stable spectral peaks or a characteristic band, a rectangle stretched by (w/R, h/R) moves those peaks to predictable frequencies, so a windowed FFT / autocorrelation of the high-pass residual of the test image estimates (w, h) BEFORE any position search, cutting the search from 4 to 2 dimensions. This is the Kutter / Pereira-Pun idea applied to an accidental template. Likely failure: the VAE round trip removes or re-generates high frequencies, and stretched rectangles larger than model resolution push the pattern to low frequencies where image content dominates.
- Phase correlation / Fourier-Mellin against Emb(y,m) - y is unlikely to work given that plain local correlation already failed after the VAE round trip (inference from our own measurement).
- Widening the basin by learning (inference, 1 to 2 h): fine-tune a COPY of the extractor (last blocks only) on simulator data with deliberately mis-registered crops (+-15% scale, +-10% shift, plus VAE round trip) and the known-message BCE loss. A decoder that tolerates mis-registration converts the needle into a hill that a coarse grid can find; the original decoder is then used only for the final refinement.

### Gaps
- No source found that compares ESS, Bayesian optimisation and CMA-ES on needle-like box objectives; the ranking above is reasoning, not benchmark evidence.
- No verified source on whether TrustMark / StegaStamp / PixelSeal residuals have stable spectral peaks.
- Fourier-Mellin registration (Reddy and Chatterjee 1996) was not fetched; no URL verified.

## Scan statistics, change-point and steganalysis payload location

### Takeaway
Scan-statistic theory says a rectangle of weak mean shift is best detected by maximising a normalised box sum over all rectangles at multiple scales, and this is computable exactly on a 128x128 map; steganalysis payload location relies on averaging residuals over many images with the same payload positions, which we do not have, but the same averaging can be done over the 100 to 256 known bits.

### Cited Findings
- For detecting a rectangle of activation in a noisy d-dimensional grid, exact asymptotic level and power are derived for oracle, multiscale and adaptive scans; an approximate scan runs in time near-linear in the grid size with the same asymptotic power as the adaptive scan — [Sharpnack and Arias-Castro, Electronic Journal of Statistics 2016](https://projecteuclid.org/euclid.ejs/1473685450)
- The plain scan is optimal only for the smallest signals; the average likelihood ratio is optimal at all scales except the smallest; a scan with scale-dependent critical values (penalised by window size) is optimal across scales — [Chan and Walther, Detection with the scan and the average likelihood ratio, 2011](https://ar5iv.labs.arxiv.org/html/1107.4344)
- Ker (MM&Sec 2008) locates LSB-replacement payload from weighted-stego residuals, assuming MANY stego images with payload at the same locations; later work improves it with better (MAP) cover estimators and extends it to LSB matching — [DBLP record](https://dblp.uni-trier.de/rec/conf/mmsec/Ker08.html); [Springer EURASIP JIVP 2020 survey text](https://link.springer.com/doi/10.1186/s13640-020-00542-2)
- Quach (Sandia) works on payload location and extraction with cover estimation — [OSTI record, Extracting hidden messages in steganographic images, 2014](https://www.osti.gov/biblio/1142884) (record only, content not read)

### Inferences
- Exact max-sum rectangle (regime 2, also rectangular holes): given any per-pixel score map s at 128x128 (segmentation logits, known-message CAM of Q5, or the gradient map of Q1), compute argmax over ALL rectangles of sum_R (s - tau) / |R|^gamma. With a 2D prefix sum this is 2D Kadane, O(n^3) = 2e6 operations per image, or brute force over the 6.8e7 boxes on GPU. gamma = 0 with tau = mean of the two class means is the likelihood-ratio box for Gaussian scores; gamma = 0.5 is the classical normalised scan. Calibrate tau and gamma on simulated rectangles. The output is a sharp axis-aligned rectangle from a blurry map, which is exactly the regime-2 prior. Take the top 3 boxes as CMA-ES starts (Q3).
- Per the Chan-Walther result, use size-dependent thresholds: small boxes need a higher normalised score than large ones, otherwise the scan prefers tiny noise blobs. Practically: penalise by sqrt(2 log(N^2/|R|)) where N^2 is the map size.
- Rows-and-columns CUSUM (inference): for an axis-aligned rectangle, the row sums and column sums of s are 1D signals with two change points each. Fit a piecewise-constant 2-change-point model to each projection (exhaustive over 128^2/2 pairs) to get an initial box in microseconds; much higher SNR than per-pixel thresholding because each projection averages 128 pixels. Fails when the rectangle is under about 5 px thick in one direction (projection contrast too small) or when there are several objects.
- Ker's principle transfers as "average over bits instead of over images" (inference): any per-location, per-bit agreement signal (Q5) should be averaged over all 100 to 256 known bits with weights by per-bit reliability, a matched subspace detector with the message as the known signature. The gain is about sqrt(n_bits) in SNR over a single-channel presence map.
- For regime 1 holes are object-shaped, not rectangular: use the scan only to decide WHERE (best box), then take the SAM / superpixel segment(s) or thresholded component inside the box as the mask.

### Gaps
- Ker 2008 and Quach full texts were not read; details of the estimators are unverified here.
- No source found on scan statistics applied to watermark presence maps.

## Learning from a perfect simulator, message conditioning and per-image test-time adaptation

### Takeaway
Published localised watermarking systems get sharp masks by reading PER-PIXEL bit predictions before pooling (WAM) or by training the decoder with masks (MaskMark); the cheapest analogue for frozen extractors is to un-pool the last layer and project onto the known message, then use the simulator on the test image itself to fit a tiny per-image head.

### Cited Findings
- WAM's extractor outputs, for every pixel, a vector of 1 + nbits values (watermark presence plus message bits); regions are found by DBSCAN over per-pixel bit strings; reported detection of watermarked areas is above 85% mIoU when 32-bit messages sit in 10% of the image — [Sander, Fernandez, Durmus, Furon, Douze, ICLR 2025](https://arxiv.org/abs/2411.07231)
- MaskMark's decoder contains a localisation module and is trained with masks applied before extraction; the authors claim better local-watermark results than WAM with 20 h of training on one A6000 — [Hu et al., NeurIPS 2025](https://arxiv.org/abs/2504.12739)
- EditGuard obtains its tamper mask by comparing the pre-defined (known) watermark with the decoded one per pixel; OmniGuard had to retrain its localisation networks when paired with a pre-trained TrustMark — [EditGuard, 2023](https://arxiv.org/pdf/2312.08883); [OmniGuard CVPR 2025 supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Zhang_OmniGuard_Hybrid_Manipulation_CVPR_2025_supplemental.pdf)
- Class activation mapping: for a network ending in global average pooling plus a linear layer, applying the linear layer at each spatial position before pooling yields an exact spatial decomposition of each logit — [Zhou et al., CVPR 2016](https://arxiv.org/abs/1512.04150) (memory); Grad-CAM generalises it to arbitrary heads — [Selvaraju et al., 2016](https://arxiv.org/abs/1610.02391) (memory)

### Inferences
- Known-message CAM (highest priority, about 30 minutes). Hook the feature map F (C x h x w) just before the extractor's global pooling. If the head is linear, compute per-location bit logits L_i(p) = w_i . F(p) + b_i, then the matched map S(p) = sum_i (2m_i - 1) * L_i(p). If the head is an MLP or flatten + dense (StegaStamp), use gradient-times-activation at the last conv with the scalar target s = sum_i (2m_i-1)*logit_i and NO channel averaging (sum over channels of F * ds/dF). This differs from what failed: it is attribution at the last feature map with the known message as the target, not input-pixel gradients and not occlusion. Run at 2 to 3 input scales and with the input upsampled 2x before the extractor's internal resize is bypassed (call the trunk directly on a larger tensor if it is fully convolutional) to get a finer h x w.
- To check in code (memory, unverified): VideoSeal / PixelSeal extractors descend from WAM and, as far as I recall, produce a per-pixel (1 + nbits) map that is then averaged. If that is true in the checkpoint used, the pre-average tensor IS a WAM-style per-pixel bit map and S(p) is available for free. Five minutes to confirm by reading the extractor's forward().
- Message-conditioned segmenter (inference): feed S(p) (and per-bit agreement summarised as mean, min-quantile and fraction of agreeing bits) as extra input channels to the existing segmentation head, instead of raw early features. The head then only has to denoise and sharpen a signal that is already message-specific, which needs far less data.
- Paired training (inference): generate (image with watermark in region, same image without) through the SAME edit + VAE round trip with a fixed seed and train on the difference of features; this removes content as a nuisance. Only possible in simulation, so use it for pretraining the head, with region sizes sampled from the test area prior (1 to 6%) and hard-example mining on regions under 2%.
- Per-image test-time fitting (inference, strongest use of the simulator). For test image y in regime 1: create K = 16 to 32 variants where random object-shaped patches (SAM segments or random blobs, 1 to 5% area) are replaced by a watermark-free version (VAE-regenerated or re-embedded with a random message, both variants), giving exact labels on the same content and the same message. Fit a tiny head (1x1 conv / logistic regression on [S(p), a few features]) on those, apply to the original y. For regime 2: paste Emb at random rectangles of the clean-looking image and do the same. Seconds per image on a GPU. Failure mode: synthetic holes differ from real ones (real holes went through the same VAE as the rest, synthetic ones are fresh), so keep the head low-capacity and feature set message-specific.
- Pseudo-label variant: fit a 2-component Gaussian mixture to S(p) within the image; the minority component is the mask. This self-calibrates the threshold per image and is immune to between-image shifts in watermark strength.

### Gaps
- Whether the specific TrustMark / StegaStamp / PixelSeal checkpoints end in pooling + linear (exact CAM) or flatten + dense was not verified from code in this session.
- No source found that does per-image test-time adaptation for watermark localisation.

## Optimising per-image mean IoU over both classes with a tiny minority class

### Takeaway
With a 1 to 6% minority class, an empty prediction scores about 0.48 to 0.50 while any overlapping blob scores above 0.5, so always predict a blob and use a per-image threshold below 0.5 tied to the expected IoU; Lovász losses optimise the per-image Jaccard directly if a head is retrained.

### Cited Findings
- The Lovász hinge / Lovász-Softmax is a tractable surrogate for the Jaccard index based on the Lovász extension of submodular losses; the paper reports better Jaccard than cross-entropy on Pascal VOC and Cityscapes and explicitly compares optimising the index per image versus over the dataset — [Berman, Rannen Triki, Blaschko, CVPR 2018](https://arxiv.org/abs/1705.08790)
- The F1-optimal decision threshold on calibrated probabilities equals half the optimal F1 value, so it is below 0.5 whenever the achievable F1 is below 1 — [Lipton, Elkan, Naryanaswamy, 2014](https://arxiv.org/abs/1402.1892) (memory)

### Inferences
- Arithmetic of the metric (mine). Minority area fraction a, predicted blob with minority IoU q: mean IoU is about (1 + q)/2 minus a small term of order a; predicting nothing gives (1 - a)/2 + 0 = about 0.47 to 0.495 (unless the scorer treats an absent class specially; check the scoring code). A false blob of area a costs at most about a/2. Hence: never output an empty minority mask; if uncertain, output the single best blob.
- Threshold rule (derived, consistent with the F1 result via F = 2J/(1+J)): adding a pixel with probability p raises expected IoU when p > J/(1 + J), where J is the IoU currently achievable. For J = 0.5 the threshold is 0.33, for J = 0.3 it is 0.23. Use low thresholds on calibrated maps for the minority class; calibrate on the simulator.
- Per-image threshold by predicted area: regress the minority area fraction from the global known-message margin (Q2 regime detector) or from the mean of the probability map, then take the top-a quantile of the map rather than a fixed threshold. This removes the between-image strength variation that hurts fixed thresholds.
- Which class is the minority must be decided first: global margin near full strength means regime 1 (minority = clean holes), near chance means regime 2 (minority = watermarked rectangle). A wrong regime call inverts the mask and costs nearly everything, so use a conservative cut calibrated in simulation.
- Component rescoring: extract up to 5 connected components / scan boxes, score each with a second-stage test (regime 2: known-message margin of the resampled crop against the significance threshold of Q3; regime 1: drop in margin when the component is replaced by y_anti at the balanced point), keep those that pass, and always keep the top one.
- Shape priors as post-processing: regime 2 masks should be exactly a rectangle (scan box), regime 1 masks should snap to SAM segments; both remove boundary blur, which dominates IoU loss for objects of 1 to 2% area (a 2 px boundary error on a 13x13 object at 128 resolution already costs about 30 to 40% IoU).

### Gaps
- The exact scorer behaviour when a class is absent from ground truth or prediction is unknown to me; it changes the "never predict empty" rule only if absent classes are scored as 1.
- No Kaggle write-up was fetched; the component-rescoring and area-quantile tricks are common practice from memory, without a verified URL.
