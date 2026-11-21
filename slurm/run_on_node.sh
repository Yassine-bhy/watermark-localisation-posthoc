#!/bin/bash
# Run a command on the team's GPU node (the running wm_step2 job), on the next free GPU (one GPU per command).
#   slurm/run_on_node.sh <name> "<command>"
# Example:  slurm/run_on_node.sh eq_train "cd /p/scratch/training2666/neves2/BlitzTeam/<task folder> && python train.py"
# The command runs as benelhajyahia1 with CUDA_VISIBLE_DEVICES set to its GPU; cd into your own folder first.
# Its output goes to benelhajyahia1/output/<jobid>_<time>_<name>.log  (see: slurm/node_status.sh)
set -e
Q=/p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1/slurm/queue
[ $# -eq 2 ] || { echo 'usage: run_on_node.sh <name> "<command>"'; exit 1; }
f=$Q/$(date +%s%N)_$1.cmd
echo "$2" > "$f"; chmod g+rw "$f" 2>/dev/null || true
echo "queued: $(basename "$f")  -> starts on the next free GPU within ~30 s"
