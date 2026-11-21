#!/bin/bash
# Run the commands listed in file $1 on the node's 4 GPUs: worker g runs lines g, g+4, g+8, ... one after another.
# Each command's output goes to output/<jobid>_c<line>.log
CMDS=$1; N=$(wc -l < "$CMDS")
for g in 0 1 2 3; do
  ( i=0; while IFS= read -r cmd; do
      if [ $((i % 4)) -eq $g ]; then echo "[gpu$g] $cmd"; CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" > output/${SLURM_JOB_ID}_c$i.log 2>&1; echo "[gpu$g] exit $? : line $i"; fi
      i=$((i+1)); done < "$CMDS" ) &
done
wait
