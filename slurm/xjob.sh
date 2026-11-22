#!/bin/bash
#SBATCH -A training2666
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -o output/xjob_%j.out
#SBATCH -e output/xjob_%j.err
# Extra node (outside the reservation): 4 GPU workers on slurm/xqueue (not our node's queue).
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
export WM_QUEUE=slurm/xqueue WM_SKIP_PIPELINE=1
bash slurm/worker.sh
