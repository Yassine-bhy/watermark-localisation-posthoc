#!/bin/bash
# Whole step 2 in ONE node allocation (we wait in the queue once):
#  1) dry run (GPU0) + synthetic data (all GPUs)   2) 3 pretraining variants in parallel
#  3) pick the variant with the best real-train mIoU   4) fine-tune: 5 CV folds + all-data model -> test predictions
set -u
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
source ../.venv/bin/activate
if [ -z "${WM_SKIP_DRY:-}" ]; then
echo "== dry run $(date)"
CUDA_VISIBLE_DEVICES=0 slurm/dryrun.sh > output/${SLURM_JOB_ID}_dryrun.log 2>&1 \
  || { echo "DRY RUN FAILED, stopping to free the node"; exit 1; }
fi
echo "== synthetic data $(date)"
slurm/multi.sh slurm/data.cmds
[ "$(ls -d results/synth/*_* | wc -l)" -eq 12 ] || { echo "SYNTH INCOMPLETE"; exit 1; }
echo "== pretraining $(date)"
slurm/multi.sh slurm/pretrain.cmds
echo "== pick variant $(date)"
BEST=$(python - <<'PY'
import json, glob
best = None
for f in glob.glob("results/log_*_pretrain.jsonl"):
    tag = f.split("log_")[1].rsplit("_pretrain", 1)[0]
    if tag == "dryrun": continue
    ev = [json.loads(l) for l in open(f) if "real_train" in l]
    if ev and (best is None or ev[-1]["real_train"]["all_at_05"] > best[0]): best = (ev[-1]["real_train"]["all_at_05"], tag)
print(best[1])
PY
)
echo "best variant: $BEST"; echo "$BEST" > results/best_variant.txt
echo "== fine-tuning $(date)"
: > output/${SLURM_JOB_ID}_ft.cmds
for f in 0 1 2 3 4 -1; do echo "python train.py finetune --init results/models/${BEST}_pre.pt --fold $f --steps 1500 --bs 16 --tag $BEST" >> output/${SLURM_JOB_ID}_ft.cmds; done
slurm/multi.sh output/${SLURM_JOB_ID}_ft.cmds
echo "== done $(date)"
