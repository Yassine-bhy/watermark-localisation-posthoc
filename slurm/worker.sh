#!/bin/bash
# Keeps the node: first the step 2 pipeline, then 4 GPU workers that each take the next command from
# slurm/queue/*.cmd (one command per file, oldest first), run it, and move it to slurm/queue/done/.
# Add work at any time:  echo "python train.py ..." > slurm/queue/$(date +%s)_name.cmd
# Stop the job cleanly:  touch slurm/queue/STOP   (workers finish their current command, then exit)
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
source ../.venv/bin/activate
slurm/pipeline.sh
Q=slurm/queue
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
