#!/bin/bash
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --job-name=wm_step2
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
# Whole node (4 GPUs): whole step 2 pipeline (slurm/worker.sh).
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
source ../.venv/bin/activate
slurm/worker.sh
