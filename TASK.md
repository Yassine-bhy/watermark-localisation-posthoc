# Task: Localize Image Watermark — European AI Championship (Paris, 21–22 Nov 2026)
Supervisors: Adam Dziedzic and Franziska Boenisch. Developed by: Simon Hector, Michel Meintz, Louis Kerner, Maitri Shah.
(Official task description, as given by the organisers.)

## Goal
The goal of this task is to find which parts of an image carry a watermark after image alterations have overwritten
or added the watermark in parts of images. Every image in the dataset was watermarked with one of three public
post-hoc watermarking methods: PixelSeal, TrustMark, or StegaStamp in a rectangular subset of the original image.
The aspect ratio of this subset does not exceed 16:1 for PixelSeal and StegaStamp and 2:1 for TrustMark.

For every test image, participants return a binary mask:
- 1 = watermarked region
- 0 = clean (unwatermarked) region

## Challenge
Some watermarking models, like Watermark Anything, are explicitly trained for localization: they tell you which
pixels are watermarked. Most post-hoc models, like PixelSeal, TrustMark and StegaStamp, are only trained to decode
one message from the whole image. The question of this task is: can we localize existing post-hoc watermarks?

What makes it hard is that these watermarking methods are not inherently trained for localization. The edits range
from large (half of the image) to small and irregular (a thin object, a face), which requires your solution to be
adaptive and general. The three watermarks behave very differently, so there is likely no one-size-fits-all solution.

Without such capabilities, later edits are attributed to the original watermark owner - potentially allowing framing
of the watermark owner by 3rd parties via the addition of undesirable content.

## Evaluation metric
Mean Intersection over Union (mIoU) over the two classes, watermarked (1) and clean (0), computed per image and then
averaged over all test images.

Mask resolution: images are 512 x 512. Masks are evaluated on a 128 x 128 grid (one cell = 4 x 4 pixels). A
ground-truth cell is 1 if at least 50% of its pixels are watermarked, otherwise 0. The starter kit contains the exact
downsampling function.

Per image:
    IoU_c = |P_c ∩ G_c| / |P_c ∪ G_c|   for c in {0, 1}   (if P_c ∪ G_c is empty, IoU_c = 1)
    mIoU_image = (IoU_0 + IoU_1) / 2
    Score = mean of mIoU_image over all test images (0..1, higher is better)
P = predicted mask, G = ground truth, P_c / G_c = cells labelled c. Both false positives and false negatives enlarge
the union. Every image counts equally, regardless of how big its edit is.

## Dataset
1,530 PNG images, split equally across the three methods.
1. Train: 510 images (170 per method): image, ground-truth mask (512 x 512 and 128 x 128), method, embedded message.
2. Test: 1,020 images (340 per method): image, method and embedded message.

Types of image edits:
- **Outpainting**: a watermarked image is extended to a larger image. The full watermark remains present but only
  covers a part of the image.
- **Object removal/alteration**: an undesired object on a watermarked image is modified or edited out. A hole in the
  watermark remains.
- **Object addition**: something unwatermarked is added to a watermarked image, covering the watermark in that
  region. Again a hole is formed.
- **Object transplantation**: an object from a watermarked image is added to an unwatermarked image. The watermark
  is only present on this object.

## Resources
- Starter kit: https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris
  (data loading, unified wrappers for PixelSeal / TrustMark / StegaStamp extractors and embedders with
  decode(images) -> bits and intermediate feature maps, bit accuracy, mask downsampling, local mIoU, submission_template.py)
- JURECA setup for all tasks: `git clone https://huggingface.co/datasets/SprintML/hackathon` (README.md)

## Submission format
A .pt file: `{"image_names": names, "masks": masks}`, names[i] ↔ masks[i], masks a tensor [num_images, 128, 128].
- each image name exactly once, no duplicates, entries 0 or 1
- file size ≤ 20 MB
Invalid submissions are rejected automatically.

## Scoring and leaderboard
- Public leaderboard: 30% of the test images. Private leaderboard (final ranking): the remaining 70%.
- Leaderboard: http://34.147.229.55/leaderboard_page — shows the best result per team only. Each request returns the
  score of that submission; if lower than the saved one, the leaderboard is not updated.

## References
- Tomáš Souček et al. "Pixel Seal: Adversarial-only training for invisible image and video watermarking." https://arxiv.org/abs/2512.16874
- Tu Bui, Shruti Agarwal, John Collomosse. "TrustMark: Robust Watermarking and Watermark Removal for Arbitrary Resolution Images." ICCV 2025.
- Matthew Tancik, Ben Mildenhall, Ren Ng. "StegaStamp: Invisible Hyperlinks in Physical Photographs." CVPR 2020.
- Tom Sander, Pierre Fernandez, Alain Durmus, Teddy Furon, Matthijs Douze. "Watermark Anything with Localized Messages." ICLR 2025. https://openreview.net/forum?id=IkZVDzdC8M
