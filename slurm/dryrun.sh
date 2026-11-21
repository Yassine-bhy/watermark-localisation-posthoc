#!/bin/bash
# Tiny end-to-end run (separate synthetic folder): synth -> pretrain 2 short epochs -> finetune 20 steps.
export SYNTH_DIR=results/synth_dryrun
for m in pixelseal trustmark stegastamp; do python synth.py --method $m --n 16 --shard 0 || exit 1; done
python train.py pretrain --res 256 --epochs 2 --bs 8 --tag dryrun || exit 1
python train.py finetune --init results/models/dryrun_pre.pt --fold 0 --steps 20 --bs 8 --tag dryrun || exit 1
