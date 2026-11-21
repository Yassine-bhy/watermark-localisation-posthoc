#!/bin/bash
# What the team node is doing: the job, commands running (gpuN_ prefix = which GPU), waiting, and recent logs.
D=/p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
squeue -r --reservation=cispahack -h -o "%i %u %T %M %N" | grep wm_step2 || squeue -h -n wm_step2 -o "%i %u %T %M %N"
echo "-- running:"; ls $D/slurm/queue/running 2>/dev/null
echo "-- waiting:"; ls $D/slurm/queue/*.cmd 2>/dev/null | xargs -n1 basename 2>/dev/null
echo "-- free GPUs: $((4 - $(ls $D/slurm/queue/running 2>/dev/null | wc -l)))"
echo "-- latest logs:"; ls -t $D/output/*.log 2>/dev/null | head -5
