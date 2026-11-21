#!/bin/bash
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --job-name=wm_pre
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
# Step 2 pretraining on synthetic data (WAM backbone, input 256), evaluated on the 510 real train images.
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
source ../.venv/bin/activate
python train.py pretrain --res 256 --epochs 3 --bs 16 --tag wam256
