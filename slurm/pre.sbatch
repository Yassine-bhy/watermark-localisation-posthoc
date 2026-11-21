#!/bin/bash
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=wm_pre
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
# Whole node (4 GPUs): runs the commands in slurm/pretrain.cmds spread over the GPUs (see slurm/multi.sh).
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
source ../.venv/bin/activate
slurm/multi.sh slurm/pretrain.cmds
