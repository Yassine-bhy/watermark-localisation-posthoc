---
name: gpu-sharing-rule
description: "GPU budget for the watermark task as of Sun 22 Nov 05:45 — all 4 GPUs of our node + extra Slurm jobs outside the reservation (user's explicit override)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 570fb962-bf82-4c39-8685-d9b6d2fde460
  modified: 2026-11-22T04:44:17.250Z
---

As of 2026-11-22 ~05:45 the user gave full autonomy ("don't even ask for permission", "use all 4 gpus", and "outside
reservation ... launch the max ... queue a lot ... but leave our node to us"). So: all 4 GPUs of our node job
15902410 (jrc0423) for this task, plus extra jobs outside the reservation (dc-gpu-devel, max 2 h per job; dc-gpu)
running slurm/xjob.sh, which feeds from its own queue slurm/xqueue/ (never from our node's slurm/queue/).
The actual rule (help desk, confirmed by the user 2026-11-22 ~05:50): at most ONE node per team INSIDE the cispahack reservation; jobs outside the reservation are allowed. Our earlier "one job per team" reading was wrong — do not raise it again.
Earlier in the night the user capped us at 1–3 GPUs to share with teammates neves2 / bensghair.
**Why:** final push for the best private-test score by Sunday 14:00; user asleep.
**How to apply:** give an expected time for every task; check that queued jobs actually started (quota errors drop
jobs silently — slurm/requeue_watch.sh requeues them); prefer robust, CV-validated changes (no overfitting).
See [[handoff-state]].
