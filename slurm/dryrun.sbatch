#!/bin/bash
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=wm_dry
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
# Step 2 dry run: tiny synthetic set in a separate folder, a few training steps, one evaluation.
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
source ../.venv/bin/activate
export SYNTH_DIR=results/synth_dryrun
for m in pixelseal trustmark stegastamp; do python synth.py --method $m --n 16 --shard 0; done
python train.py pretrain --res 256 --epochs 2 --bs 8 --tag dryrun
python train.py finetune --init results/models/dryrun_pre.pt --fold 0 --steps 20 --bs 8 --tag dryrun
