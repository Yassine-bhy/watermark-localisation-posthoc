# R3 – Competition / organiser / prior-solution research (2026-11-22)

## Actionable summary
- No paper, blog, code or dataset-generation script for THIS image task is public. SprintML has no arXiv paper on post-hoc image-watermark localization or partial-edit forgery (checked author lists for Boenisch, Meintz, Kerner, the sprintml.github.io publications list and blog, and the sprintml GitHub org). Simon Hector and Maitri Shah have no papers. Maitri Shah ("maitri01" on HF) builds the hackathon datasets.
- HF repo `SprintML/Hackathon_Watermark_Localisation_Task` (the `_Paris` name redirects there). Files: .gitattributes, README.txt, requirements.txt, starter_kit.zip (10 KB), train.zip (191 MB), test.zip (379 MB). Two commits by user `Sihe00003`: "Watermark localisation (images) v2.1: train, test, starter kit" (2026-11-21 01:05) and "Added requirements.txt" (16:16). The commit messages name no generator, inpainting model or seeds. There are no discussions. "v2.1" means earlier internal versions existed.
- The earlier `SprintML/watermark_localization` (June 2026 Grand Finale) is the TEXT version: per-token labels, KGW/Unigram/Gumbel/TextSeal, TPR@0.1%FPR, with all keys disclosed. The best public write-up (rank 2, florianzainzinger1/cispa-hackathon-grand-finale-2026) used per-token detector signals from every scheme, a learned span-length prior (explicit-duration HSMM) to remove false positives in gaps, and a strict validation gate against the 30% public split. The image analogues: decoder-feature maps from all 3 watermarks plus a strong shape prior (rectangle or rectangle minus hole), and guarding against public-LB overfit.
- TrustMark 0.8.0 (pinned): it computes the residual at 256x256, upsamples it and feather-pastes it with feather = 1% of the region (≥1 px). So the TrustMark signal is low-frequency and its boundary is nearly sharp. The package also ships a removal net (`tm.remove_watermark`, ckpt `trustmark_rm_P/Q/B`). |x − RM(x)| is a free per-pixel TrustMark residual map to use as an extra feature or channel.
- PixelSeal (videoseal) gives no spatial output natively. Its extractor makes a per-pixel map before the global pool, so use that pre-pool map (kit `features()`).
- Closest external benchmark: arXiv 2609.16832, "What Breaks Local Watermarks?". It builds edits with a Stable Diffusion inpainting pipeline, plus outpainting that resizes the watermarked image into a larger canvas (the same as our outpainting setup). Under inpainting or outpainting, decoding of TrustMark and PixelSeal falls to near chance even outside the edit, but localization signal survives better than bit accuracy. This suggests the decoders' per-pixel features are useful even when the bits fail. Code is promised at github.com/garandorinc/markbench. IoU references: WAM 0.906 and MaskWM 0.779 under inpainting.
- No public solution from the Paris Nov-2026 image task exists on GitHub. Searches on repos pushed since 2026-11-19 returned nothing.

## Query log
1. HF API dataset/commits/discussions/tree for SprintML/Hackathon_Watermark_Localisation_Task(_Paris) [found: https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task]
2. HF API list SprintML datasets/models/spaces [found: 23 datasets; relevant: watermark_localization (text, Jun 2026), watermark_detection (Paris 2025), watermark-removal (Vienna 2025); no spaces]
3. SprintML/watermark_localization files/config [found: https://huggingface.co/datasets/SprintML/watermark_localization – LLM task, not image]
4. Web: "watermark localization outpainting inpainting PixelSeal TrustMark StegaStamp" [found: https://arxiv.org/abs/2609.16832]
5. Web: Dziedzic Boenisch watermark forgery localization partial edits [not found; related: arXiv 2610.04151 (partial-edit blobs benchmark), 2611.16368 ReliaGuard]
6. Web: Meintz / Kerner watermark [found only BitMark 2506.21209, Radioactive WM 2506.23731, Data Provenance IAR 2606.28386 – not localization]
7. arXiv API au:Boenisch AND abs:watermark [found 5 entries, none on localization]; au:Dziedzic listing [429 / not retrieved]
8. Web: "Maitri Shah" / "Simon Hector" CISPA watermark [not found]
9. GitHub org sprintml repos [found HackathonParis (2025: adv examples, image attribution, watermark detection), HackathonVienna (watermark removal), hackathonStockholm; nothing for this task]
10. sprintml.github.io publications+blog grep "watermark" [only SERUM, BitMark, radioactivity papers]
11. Web: European AI Championship Paris watermark localisation solution github [found 2025 Paris repo floriandgg/cispa (watermark detection, TPR@1%FPR) – different task]
12. GitHub search: "watermark localization hackathon", "cispa hackathon", "wmloc", "stegastamp trustmark", recent pushes [found finale text-WM repos: https://github.com/florianzainzinger1/cispa-hackathon-grand-finale-2026 , https://github.com/andresmartinezcarcel/cispa-cybersecurity-ai-hackathon-finale ; no image-localisation solutions]
13. Web: Erasing the Invisible NeurIPS 2024 winning solution [found: https://arxiv.org/abs/2508.21072 (VAE-based adaptive attack, clustering by artifacts + img2img diffusion); 5th place StegaStamp: https://arxiv.org/abs/2505.01474] – removal, not localization
14. Web: localizing post-hoc watermarks openreview [found WAM https://arxiv.org/abs/2411.07231 , block-level https://arxiv.org/abs/2512.14994 , MaskWM https://arxiv.org/abs/2504.12739 (outpainting via stable-diffusion-2-inpainting)]
15. Web: PixelSeal localization heatmap [not found; PixelSeal https://arxiv.org/abs/2512.16874 has no spatial output]
16. cispa.de Paris page [no task details]
17. TrustMark README/CONFIG + trustmark 0.8.0 source [found: 256px residual, FEATHERING_RESIDUAL=0.01, removal model trustmark_rm_*]
