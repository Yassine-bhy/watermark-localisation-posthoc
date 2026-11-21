"""Realistic synthetic data, step 2/3 (generator venv .venv_gen): SD2 inpainting of the 'gen' regions.
    .venv_gen/bin/python inp_gen.py <shard dir>
Composite: generated pixels inside the (feathered) region, the watermarked base outside. Writes inpainted.npy."""
import sys
import numpy as np, torch
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline

d = sys.argv[1]; z = np.load(f"{d}/prep.npz")
pipe = StableDiffusionInpaintPipeline.from_pretrained("sd2-community/stable-diffusion-2-inpainting", torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.set_progress_bar_config(disable=True)
base, gen, prompts = z["base"], z["gen"], z["prompt"]; out = base.copy(); g = torch.Generator("cuda").manual_seed(0)
idx = [i for i in range(len(base)) if gen[i].any()]
for j in range(0, len(idx), 8):
    b = idx[j:j + 8]
    ims = [Image.fromarray(base[i]) for i in b]; mks = [Image.fromarray(gen[i].astype(np.uint8) * 255) for i in b]
    res = pipe(prompt=[str(prompts[i]) for i in b], negative_prompt=["text, watermark, blurry"] * len(b), image=ims, mask_image=mks,
               height=512, width=512, num_inference_steps=20, guidance_scale=6.0, generator=g).images
    for i, r in zip(b, res):
        soft = np.asarray(Image.fromarray(gen[i].astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)[..., None] / 255
        soft = np.maximum(soft, gen[i][..., None])                  # generated region itself fully replaced
        out[i] = (soft * np.asarray(r, np.float32) + (1 - soft) * base[i]).round().clip(0, 255).astype(np.uint8)
    print(j + len(b), "/", len(idx), flush=True)
np.save(f"{d}/inpainted.npy", out); print("gen done")
