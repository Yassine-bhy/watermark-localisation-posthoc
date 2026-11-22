---
name: hackathon-rules-and-workflow
description: "Watermark localisation hackathon (Paris, 21–22 Nov 2026): what's allowed and how the user wants me to work"
metadata:
  node_type: memory
  type: project
  originSessionId: 5e88ecf5-8740-4cca-82d4-763a43a356f0
  modified: 2026-11-21T19:18:39.103Z
---

Any pretrained model / external data is allowed, no limits; only the score matters, fine-tune as much as helps (user, 2026-11-21).
Reservation cispahack ends 2026-11-22 14:00.

Organiser rules (Discord, 2026-11-21, clarified ~20:00): the WHOLE TEAM (everyone under /p/scratch/training2666/neves2/BlitzTeam) may have at most ONE job, running or queued. Check before any sbatch: T=$(pwd | grep -oE '/p/scratch/training2666/[^/]+/[^/]+'); squeue -r --reservation=cispahack -h -o "%u|%T|%Z" | grep "$T". Teammates share our worker node via slurm/run_on_node.sh (status: slurm/node_status.sh); keep at most 2 of OUR commands running so >=2 GPUs stay free for teammates (user, 2026-11-21). Keep the course artifact (docs/course_template.html -> build_course.py -> publish) updated after each result. Chain with --dependency=afterany, arrays only with %1. Earlier wording: request only ONE node (4 GPUs) at a time — never several jobs/array tasks running in parallel (jobs get whole nodes exclusively; pack work onto the 4 GPUs via slurm/multi.sh, chain stages with --dependency); repeat offenders lose GPU access. Leaderboard token: team-only, never commit it or put it in any repo/public place (disqualification).

User's working rules: own subfolder `benelhajyahia1/` (git, one commit per step); shared `.venv` one level up — install what's needed WITHOUT asking (user, 2026-11-21), but never break it: dry-run first, `--exclude-newer` ~2 weeks back, no upgrades of existing pins (torch 2.5.1 etc.), then just mention what was installed; heavy work via Slurm (account training2666, reservation cispahack, partition dc-gpu, gpu:1); one step at a time, explain before coding.

**Why:** user said "i can't just rely on you" — wants to verify everything visually.
**How to apply:** every step ships a notebook (built by notebooks/make_NN_*.py, executed, plus HTML sent via SendUserFile) showing images, masks, maps, predictions and the CV tables; all performance claims = held-out CV numbers.
