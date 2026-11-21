#!/bin/bash
# Keeps the node: first the step 2 pipeline, then 4 GPU workers that each take the next command from
# slurm/queue/*.cmd (one command per file, oldest first), run it, and move it to slurm/queue/done/.
# Add work at any time:  echo "python train.py ..." > slurm/queue/$(date +%s)_name.cmd
# Stop the job cleanly:  touch slurm/queue/STOP   (workers finish their current command, then exit)
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
# Run from a frozen copy of the shell scripts: bash reads scripts while running, so later edits of slurm/*.sh
# must not touch this job. (Python files are read when each command starts -> edits apply to the next command.)
if [ -z "${WM_SNAP:-}" ]; then
  S=output/${SLURM_JOB_ID:-local}_scripts; mkdir -p $S; cp slurm/*.sh slurm/*.cmds $S/
  sed -i "s#slurm/\(multi\|dryrun\|pipeline\|data\|pretrain\)#$S/\1#g" $S/*.sh
  WM_SNAP=1 exec bash $S/worker.sh
fi
source ../.venv/bin/activate
[ -z "${WM_SKIP_PIPELINE:-}" ] && bash $(dirname "$0")/pipeline.sh
Q=${WM_QUEUE:-slurm/queue}
for g in 0 1 2 3; do
  ( while [ ! -e $Q/STOP ]; do
      f=$(ls $Q/*.cmd 2>/dev/null | head -1)
      if [ -z "$f" ]; then sleep 30; continue; fi
      b=$(basename "$f"); mv "$f" "$Q/running/gpu${g}_$b" 2>/dev/null || continue     # claim (atomic mv)
      echo "[gpu$g $(date +%H:%M)] start $b"
      CUDA_VISIBLE_DEVICES=$g bash "$Q/running/gpu${g}_$b" > "output/${SLURM_JOB_ID}_${b%.cmd}.log" 2>&1
      echo "[gpu$g $(date +%H:%M)] exit $? $b"; mv "$Q/running/gpu${g}_$b" "$Q/done/$b"
    done ) &
done
wait
