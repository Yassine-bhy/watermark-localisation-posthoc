#!/bin/bash
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=wm_synth
#SBATCH --array=0-11
#SBATCH --output=output/%A_%a.out
#SBATCH --error=output/%A_%a.err
# Step 2 synthetic data: 3 methods x 4 shards x 1500 images.
METHODS=(pixelseal trustmark stegastamp)
M=${METHODS[$((SLURM_ARRAY_TASK_ID % 3))]}; SH=$((SLURM_ARRAY_TASK_ID / 3))
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
source ../.venv/bin/activate
python synth.py --method $M --n 1500 --shard $SH
