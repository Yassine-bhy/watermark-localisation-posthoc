# Provenance of the "Localize Image Watermark" hackathon dataset (SprintML, Paris, Nov 2026)

Research done 21-22 Nov 2026, about 15 tool calls. "Confirmed" means I read it in the cited source. Everything under "Inferences" is my reasoning and is not confirmed by any source. No source states which editing model, VAE, mask source or photo dataset the organisers used.

## 1. Does "What Breaks Local Watermarks?" exist, and is it the basis of the task? Any SprintML paper on this?

### Takeaway
The paper exists (arXiv 2609.16832, 15 Sep 2026) but is from the University of Edinburgh and Garandor, not SprintML/CISPA, and its setup differs from the hackathon (Stable Diffusion 1.5 inpainting only, no StegaStamp, no object edits). Its most useful result for us: SD inpainting without paste-back destroys the watermark outside the edit mask, and copying the outside pixels back restores it. I found no SprintML paper on image watermark localisation.

### Cited Findings
- Title "What Breaks Local Watermarks? A Robustness Benchmark for Local Invisible Image Watermarking"; authors Kai Yao, Bence Szilágyi, Sebestyén Kamp, Máté Poór, Máté Szilveszter, Matyas K. Zsoldos, Marc Juarez; submitted 15 Sep 2026; accepted at AISec 2026 (workshop at ACM CCS 2026) — [arXiv abs](https://arxiv.org/abs/2609.16832)
- Affiliations are University of Edinburgh and Garandor (Edinburgh, UK). No CISPA or SprintML author — [arXiv HTML](https://arxiv.org/html/2609.16832v1)
- Methods evaluated: MaskWM, WAM, OmniGuard, TrustMark, PixelSeal. StegaStamp is not evaluated. 55 transformations in four groups (signal distortions, alignment changes, indirect local edits, direct watermark edits) — [arXiv abs](https://arxiv.org/abs/2609.16832)
- Datasets: COCO 2017 and SA-1B (native segmentation masks used as watermark regions), DIV2K and MIRFLICKR-25K (procedurally sampled rectangular or irregular masks). One watermark region per image — [arXiv HTML, sec. 3.3](https://arxiv.org/html/2609.16832v1)
- TrustMark local adaptation: mask replaced by its smallest axis-aligned bounding box, crop resized to the encoder's native resolution, residual computed, per-channel spatial mean subtracted, residual resized back and added. Variant Q, 100 bits, error correction disabled — [arXiv HTML, app. A.5](https://arxiv.org/html/2609.16832v1)
- PixelSeal local adaptation: embed on the whole image, take the residual (watermarked minus original), keep it only inside the selected region. Released 256-bit model — [arXiv HTML, app. A.5](https://arxiv.org/html/2609.16832v1)
- Generative edits in the paper: "inpainting (independent region)" on a random rectangle with side ratios from 0.05 up to 0.15/0.35/0.50; "inpainting (independent region, outside restored)" which copies the watermarked pixels back outside the mask; "outpainting (fixed canvas)" which regenerates the complement of the watermark region "without restoring unmasked pixels after generation"; "outpainting (expanded canvas)" which places the image on a canvas 1.5-2.0x larger; "inpainting (watermark region)" on the watermark mask dilated by about 3% of the shorter side — [arXiv HTML, Table 4](https://arxiv.org/html/2609.16832v1)
- Finding: plain inpainting and fixed-canvas outpainting push every method below the detection threshold, while the "outside restored" variant brings MaskWM, PixelSeal and WAM back above it. The authors attribute the loss to "the whole-image reconstruction induced by the Stable Diffusion inpainting pipeline". Higher edge density in the watermark region is associated with weaker recovery — [arXiv HTML, sec. 4](https://arxiv.org/html/2609.16832v1)
- Code: https://github.com/garandorinc/markbench (repo is live, HTTP 200). The only generative attack file is `stable_diffusion_inpainting.py`, which loads `runwayml/stable-diffusion-inpainting` with diffusers `StableDiffusionInpaintPipeline`, float16, default 25 steps, empty prompt, and resizes the output back with Lanczos. No FLUX, SDXL, Kontext or LaMa — [markbench README](https://github.com/garandorinc/markbench), [attack source](https://raw.githubusercontent.com/garandorinc/markbench/HEAD/markbench/benchmarking/attacks/generative_ai/stable_diffusion_inpainting.py)
- The repo ships no datasets and no watermark weights — [markbench README](https://github.com/garandorinc/markbench)
- The hackathon task text names the "framing of the watermark owner by 3rd parties" motivation and cites only PixelSeal, TrustMark, StegaStamp and Watermark Anything. It does not cite arXiv 2609.16832 — local file `/p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1/TASK.md` (official task text as copied by the team)

### Inferences
- The hackathon uses the same TrustMark variant (Q, 100 bits) and PixelSeal model (256 bits) as this paper, and the same "restrict the watermark to a region" idea, so the organisers may have known the paper. But the two differ in watermark set, edit types and (per the team's VAE test) the generative model, so markbench is very unlikely to be the generator of the hackathon data.
- The paper's local-embedding recipes (PixelSeal: masked residual; TrustMark: crop, resize, embed, resize residual back) are plausible templates for how the organisers embedded inside a rectangle. The TrustMark 2:1 aspect limit in the task fits a crop-and-resize-to-square embedding.

### Gaps
- Not found: any 2025-2026 paper by Dziedzic, Boenisch or SprintML on image watermark localisation, watermark forgery by editing, or local watermark robustness. Two web searches returned nothing attributable to them. This is weak evidence of absence (two queries only).
- Numeric scores in the paper were lost in my text extraction (math markup stripped), so I report its findings qualitatively only.

## 2. What do the SprintML Hugging Face repos say?

### Takeaway
The repos contain no generation script and no statement about editing models, VAE or source images. The one real hint is a docstring in the starter kit: `full_watermark` is "true if the whole image was watermarked and a part of it restored to the clean original".

### Cited Findings
- `SprintML/Hackathon_Watermark_Localisation_Task_Paris` and `SprintML/Hackathon_Watermark_Localisation_Task` hold identical files (same blob hashes): `README.txt` (1.7 kB), `requirements.txt`, `starter_kit.zip` (10 kB), `train.zip` (191 MB), `test.zip` (379 MB). No dataset card (`README.md` returns "Entry not found") — [HF tree API](https://huggingface.co/api/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris/tree/main)
- The repo was created 20 Nov 2026 and last modified 21 Nov 2026 16:16 UTC — [HF API](https://huggingface.co/api/datasets/SprintML/Hackathon_Watermark_Localisation_Task)
- `README.txt` describes only the starter kit and the submission format. It refers to a separate "participant document" for task description, data and scoring — [README.txt](https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris/raw/main/README.txt)
- `requirements.txt`: torch 2.5.1, onnxruntime-gpu 1.23.2 (StegaStamp), `trustmark==0.8.0`, videoseal cloned by `download_models.py`. `pycocotools` is listed under the comment "PixelSeal: ... its dependencies", so it is a videoseal dependency and not evidence of COCO as source — [requirements.txt](https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris/raw/main/requirements.txt)
- Starter-kit docstring: train metadata has `full_watermark`, "true if the whole image was watermarked and a part of it restored to the clean original" — local copy `/p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/wmloc/data.py`
- Train metadata keys are `image_name, method, message, full_watermark`; test keys are `image_name, method, message`. Train has exactly 85 images per (method, full_watermark) cell. The PNGs carry no text chunks (PIL `info` is empty) — local files under `/p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/release/`
- `SprintML/hackathon` contains only JURECA setup material (`README.md`, `hackathon_setup.sh`, `teammate.sh`) — [README](https://huggingface.co/datasets/SprintML/hackathon/raw/main/README.md)
- Other SprintML repos with "watermark" in the name: `watermark_localization` (July 2026) is an LLM text watermark task (config says `task: llm_watermark_localization`, TextSeal/KGW/Unigram); `watermark_detection` and `watermark-removal` (March 2026) are earlier image tasks with a `Dataset.zip` and a template only — [watermark_config.yaml](https://huggingface.co/datasets/SprintML/watermark_localization/raw/main/watermark_config.yaml), [HF org listing](https://huggingface.co/api/datasets?author=SprintML&limit=100)

### Inferences
- "A part of it restored to the clean original" reads like a mask composite: `final = mask * watermarked + (1 - mask) * clean`. If taken literally, the hole in a "full" image holds pixels of a clean (unwatermarked) reference image, and the edit itself may have been made before watermarking (for example on a clean before/after edit pair), with the watermark then applied to the edited image and removed inside the object mask. That would be a different pipeline from "watermark first, then run an inpainting model over it".
- This literal reading is in tension with the team's observation that the surviving watermark is weakened as after a FLUX VAE round trip. Both can hold if the composite was done first and the whole composite then went through a FLUX-family model or VAE, or if the docstring is a loose description. I could not settle this.
- The exact 85/85 balance per method and the four named edit types suggest a scripted generator with a fixed quota per edit type, not a scraped dataset.

### Gaps
- The "participant document" is not in the HF repos. The team's copy (`TASK.md`) has no generation details beyond the four edit types.
- I did not open `starter_kit.zip` from HF separately; I relied on the unpacked local copy (`wmloc/`), which I assume is the same.
- `SprintML/watermark_detection` and `watermark-removal` were not inspected for reused source images.

## 3. Which source dataset with object masks and these four edit categories matches best?

### Takeaway
Not determined. No public dataset I found uses the category set "outpainting / object removal or alteration / object addition / object transplantation", and no source names the hackathon's photo dataset.

### Cited Findings
- The four edit types are defined by the organisers as: outpainting (watermarked image extended to a larger image), object removal/alteration (hole in the watermark), object addition (something unwatermarked added, hole), object transplantation (object from a watermarked image added to an unwatermarked image) — local `TASK.md` (path above)
- The closest related benchmark uses COCO 2017 and SA-1B segmentation masks as region sources — [arXiv 2609.16832](https://arxiv.org/html/2609.16832v1)
- PixelSeal was trained on SA-1B and evaluated on SA-1B photos and Meta AI images — [Pixel Seal paper, 2025](https://arxiv.org/html/2512.16874)
- A search for a dataset combining these four categories returned only general editing/forensics benchmarks (NeXT-IMDL arXiv 2512.23374 with removal/replacement/addition and FLUX-based editors; LocateEdit-Bench arXiv 2602.05577; ICE-Bench with outpainting, local subject addition and removal; CompBench). These are search-snippet level findings; I did not open the papers — [NeXT-IMDL](https://arxiv.org/pdf/2512.23374v1), [LocateEdit-Bench](https://arxiv.org/html/2602.05577v1)
- The team's handoff note lists "COCO retrieval" among dead ends — local `/p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1/HANDOFF.md`

### Inferences
- "Object transplantation" needs an object mask on a watermarked donor image and a second clean image; "removal" and "addition" need object-shaped masks. Any dataset with instance masks (COCO, SA-1B, OpenImages) or SAM-generated masks would serve. The object-shaped holes the team sees are consistent with either.
- If the team's COCO retrieval attempt covered only val2017, it does not exclude COCO train2017 or other sets.

### Gaps
- MagicBrush, PIPE, SEED-Data-Edit, AnyEdit, TGIF, GIM, SAGI-D, CocoGlide and BR-Gen were not checked individually (time box). I cannot say which matches best.
- Source photo dataset: not found.

## 4. Do common inpainting pipelines return the unmasked region pixel-exact or through the VAE?

### Takeaway
In diffusers, none of the checked pipelines paste the original pixels back by default: the whole output is a VAE decode. `FluxFillPipeline` and `FluxKontextPipeline` have no paste-back option at all; the SD, SDXL and `FluxInpaintPipeline` only overlay the original when `padding_mask_crop` is set.

### Cited Findings
(all from diffusers `main`, read 22 Nov 2026)
- `FluxFillPipeline`: VAE-encodes the masked image as conditioning, ends with `self.vae.decode(latents)`. The file contains no `padding_mask_crop`, no `apply_overlay` and no per-step latent blending (`init_latents_proper`) — [pipeline_flux_fill.py](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/flux/pipeline_flux_fill.py)
- `FluxKontextPipeline`: no mask handling, ends with `self.vae.decode(latents)` — [pipeline_flux_kontext.py](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/flux/pipeline_flux_kontext.py)
- `FluxInpaintPipeline`: blends latents each step (`latents = (1 - init_mask) * init_latents_proper + init_mask * latents`), decodes the full image, and calls `apply_overlay(mask_image, original_image, ...)` only `if padding_mask_crop is not None` (default `None`) — [pipeline_flux_inpaint.py](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/flux/pipeline_flux_inpaint.py)
- `FluxKontextInpaintPipeline`: same per-step latent blending and full decode; my grep found a `padding_mask_crop` argument but no `apply_overlay` call in the file — [pipeline_flux_kontext_inpaint.py](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/flux/pipeline_flux_kontext_inpaint.py)
- `StableDiffusionInpaintPipeline` and `StableDiffusionXLInpaintPipeline`: latent blending only when the UNet has 4 input channels (`if num_channels_unet == 4`), i.e. not for the dedicated 9-channel inpainting checkpoints; full VAE decode; `apply_overlay` only `if padding_mask_crop is not None` — [SD inpaint](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py), [SDXL inpaint](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py)
- `apply_overlay` pastes the original image over the output using the inverted mask, which is a pixel-exact paste outside the mask — [image_processor.py](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/image_processor.py)
- Empirical confirmation for SD 1.5 inpainting: without paste-back, watermark recovery outside the edit collapses; with paste-back it returns — [arXiv 2609.16832](https://arxiv.org/html/2609.16832v1)

### Inferences
- With `FluxFillPipeline` (and the 9-channel SD/SDXL inpainting checkpoints) the unmasked region is not a plain VAE round trip: it is generated by the denoiser conditioned on the masked-image latents and then decoded. With latent-blending pipelines (`FluxInpaintPipeline`, Kontext inpaint) the unmasked region is close to encode-then-decode of the input. Either way, a pure VAE encode/decode of the fresh watermark is a lower bound on the damage; FLUX Fill output may differ slightly more from the input than a pure round trip does.
- The team's observation (weakened but still present watermark, best matched by a FLUX VAE round trip, edit rectangles not on an 8/16 px grid) fits a FLUX-family pipeline run without paste-back. Rectangles off the 16 px grid are compatible with FLUX Fill because the mask is supplied at pixel resolution (packed into the conditioning), and also with any rescaling of the canvas after generation. This remains unconfirmed.
- For outpainting ("partial, exact rectangle"), the watermarked rectangle would then also have passed through the generative model, which explains why it is weaker than a fresh embedding.

### Gaps
- PowerPaint, BrushNet and LaMa were not checked (time box). LaMa has no VAE (it is a feed-forward CNN), but I did not verify its default compositing behaviour from source.
- Behaviour of ComfyUI/hosted FLUX Fill or Kontext APIs (which may composite) was not checked.
- Whether the final images were resized to 512x512 after generation is unknown; a resize would also weaken pixel-exact patterns.

## 5. Known bit-accuracy numbers after VAE round trip or regeneration

### Takeaway
I found no published bit accuracy for PixelSeal, TrustMark or StegaStamp after a FLUX VAE round trip. The PixelSeal paper does not test any generative or VAE attack. The only directly relevant evidence is qualitative (arXiv 2609.16832) and snippet-level claims about SD-based regeneration.

### Cited Findings
- Pixel Seal paper: attacks are valuemetric, geometric, compression and a combined JPEG+crop+brightness; no VAE, diffusion regeneration or inpainting evaluation. Bit accuracy on SA-1B photos: identity 1.00, valuemetric 0.98, compression 0.99, geometric 0.93, combined 0.94. TrustMark-Q on Meta AI images: identity 1.00, valuemetric 0.97, compression 1.00, geometric 0.67, combined 0.54 (as extracted by a summarising fetch; check Table 1 before quoting) — [arXiv 2512.16874](https://arxiv.org/html/2512.16874)
- Pixel Seal embeds at 256x256, bilinearly upsamples the watermark to the input resolution and multiplies by a JND map; 256 bits; PSNR 48.9 dB on Meta AI images — [arXiv 2512.16874](https://arxiv.org/html/2512.16874)
- SD 1.5 inpainting without paste-back and fixed-canvas outpainting leave TrustMark and PixelSeal (and all others) below the paper's detection threshold; the abstract says generative local edits "can completely impair payload recovery" — [arXiv 2609.16832](https://arxiv.org/abs/2609.16832)
- Search-snippet claim, paper not opened: a Stable Diffusion based editing attack pushes detection rates below 5% for StegaStamp and TrustMark — [arXiv 2510.05978](https://arxiv.org/html/2510.05978v1)
- WAVES (ICML 2024) evaluates StegaStamp, Stable Signature and Tree-Ring under distortion, regeneration (diffusion and VAE) and adversarial attacks, reporting TPR at 0.1% FPR — [arXiv 2401.08573](https://arxiv.org/html/2401.08573v3)

### Inferences
- The hackathon watermark being "weaker but usable" after the suspected FLUX pass, while SD 1.5 inpainting wipes PixelSeal and TrustMark out, is consistent with the 16-channel FLUX VAE reconstructing fine detail far better than the 4-channel SD 1.5 VAE. This supports the team's FLUX-over-SD/SDXL conclusion but is not a published number.

### Gaps
- WAVES per-attack numbers for StegaStamp (VAE regeneration, diffusion regeneration): not retrieved.
- TrustMark paper numbers under regeneration: not retrieved.
- "Invisible image watermarks are provably removable" numbers: not retrieved.
- No published FLUX VAE round-trip numbers for any of the three methods found.

## 6. What this implies for faithful synthetic training data

### Takeaway
Nothing in public sources pins down the generator, so the synthetic pipeline should be validated against the 510 real train images. The strongest sourced guidance is: embed locally the way arXiv 2609.16832 does, run the whole image through a generative pass without paste-back, and cover both "watermark then edit" and "composite with clean original" hole semantics.

### Cited Findings
- Whole-image reconstruction, not the edit itself, is what degrades the watermark outside the mask; paste-back removes the effect — [arXiv 2609.16832](https://arxiv.org/html/2609.16832v1)
- Default diffusers inpainting pipelines do not paste back — sources in section 4
- The organisers describe "full" images as whole-image watermark with a part "restored to the clean original" — local `wmloc/data.py`
- Local embedding recipes for TrustMark-Q (bounding box crop, resize, residual, mean-subtract, resize back) and PixelSeal (masked global residual) — [arXiv 2609.16832, app. A.5](https://arxiv.org/html/2609.16832v1)

### Inferences
- Recipe candidates worth generating and comparing on real-data CV, in order of fit to the evidence: (a) watermark rectangle, FLUX Fill inpaint/outpaint with object or border mask, no paste-back; (b) same but latent-blended (`FluxInpaintPipeline`) or plain FLUX VAE round trip after a pixel composite, which is what the team already uses; (c) composite watermarked and clean versions by mask with no generative pass on part of the data, in case some edit types were built that way.
- For "full" images, the hole content could be either generated pixels or clean-original pixels. Synthetic holes filled with clean original content (identical scene, no watermark) are the harder case and are directly supported by the docstring, so they should be included.
- Edge handling matters: if the real pipeline has no paste-back, there is no sharp compositing seam at the mask boundary, so synthetic data with hard pixel composites applied after the VAE pass would teach a seam cue that the real data lacks. Compositing before the VAE pass avoids this.
- Object masks from COCO instances or SAM are a reasonable stand-in for the unknown mask source; mask area statistics should be matched to the team's measured medians (holes about 6% of area, partial regions 1-4%).
- TrustMark regions should be embedded by crop-and-resize with aspect at most 2:1; PixelSeal and StegaStamp up to 16:1, per the task text.

### Gaps
- Editing model, VAE, mask source and photo dataset used by the organisers: not found in any public source. All four remain inference.
- Whether edits were applied before or after watermarking is unresolved (see section 2).
