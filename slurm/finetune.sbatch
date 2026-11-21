#!/bin/bash
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=wm_ft
#SBATCH --array=0-5
#SBATCH --output=output/%A_%a.out
#SBATCH --error=output/%A_%a.err
# Step 2 fine-tuning on real images: tasks 0-4 = 5-fold CV (held-out fold), task 5 = all 510 -> test predictions.
FOLD=$SLURM_ARRAY_TASK_ID; [ "$FOLD" -eq 5 ] && FOLD=-1
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
source ../.venv/bin/activate
python train.py finetune --init results/models/wam256_pre.pt --fold $FOLD --steps 1500 --bs 16 --tag wam256
