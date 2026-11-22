---
name: standard-baseline-first
description: "ML competitions: build the standard strong baseline (U-Net + pretrained encoder + standard augmentations + seeds/TTA/ensembles) before any custom idea; ablate augmentations instead of reasoning them away"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 570fb962-bf82-4c39-8685-d9b6d2fde460
  modified: 2026-11-22T13:24:37.735Z
---

Lesson from the watermark-localisation hackathon (Nov 2026): we finished 0.739 while two teams reached 0.80–0.81 with
textbook methods — a full U-Net whose encoder is the pretrained decoder, random resized-crop / shift augmentation
(which happened to match the real data's ~1.03 zoom after embedding), many seeds, TTA, ensembles, a learned per-image
selector. I started from a frozen probe + small head, theorised that resize/flip augmentation would "break the
watermark" instead of testing it, anchored on a guessed degradation (FLUX VAE) without measuring the real transform,
and spent hours on exotic signals before the plain supervised segmenter was strong.
The user was frustrated that a non-ML team using a model did better by just following standard practice.
**Why:** standard recipes are strong defaults; custom reasoning without ablations hid the decisive factor.
**How to apply:** first hours of any ML task — standard strong architecture + standard augmentations, ablate each
augmentation (one short run each), measure the real data's processing directly (e.g. interpolation artefacts,
statistics vs synthetic), then custom ideas. Periodically ask "how far are we from a strong standard pipeline?".
The full playbook is the user-level skill ~/.claude/skills/ml-competition-playbook (invoke it at the start of any ML competition). See [[handoff-state]].
