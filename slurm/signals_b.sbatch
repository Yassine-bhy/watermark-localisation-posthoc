#!/bin/bash
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=wm_sigb
#SBATCH --array=0-5
#SBATCH --output=output/%A_%a.out
#SBATCH --error=output/%A_%a.err
# Step 1b signal maps: one array task per (method, split).
METHODS=(pixelseal trustmark stegastamp); SPLITS=(train test)
M=${METHODS[$((SLURM_ARRAY_TASK_ID % 3))]}; S=${SPLITS[$((SLURM_ARRAY_TASK_ID / 3))]}
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
source ../.venv/bin/activate
python signals_b.py --method $M --split $S
