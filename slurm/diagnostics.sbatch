#!/bin/bash
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=wm_diag
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
# Step 1 diagnostics (CPU work; runs after the signal maps exist).
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
source ../.venv/bin/activate
python diagnostics.py
