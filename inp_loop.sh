#!/bin/bash
# All-night generator of realistic synthetic data: for shard k = 0.. and each method: prep -> inpaint -> FLUX round trip.
# Stops after MAXK shards per method, or when results/synth_inp/STOP exists.
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
export PYTHONUNBUFFERED=1 HF_HOME=/p/scratch/training2666/neves2/BlitzTeam/.cache/huggingface HF_HUB_OFFLINE=1
MAXK=${MAXK:-15}
for k in $(seq 0 $((MAXK - 1))); do
  for m in trustmark stegastamp pixelseal; do
    [ -f results/synth_inp/STOP ] && exit 0
    d=results/synth_inp/${m}_$k
    [ -f $d/images.npy ] && continue
    for t in 1 2 3; do
      ../.venv/bin/python inp_prep.py --method $m --shard $k --n 400 && .venv_gen/bin/python inp_gen.py $d && ../.venv/bin/python inp_final.py $d && break
      echo "retry $t $d"; sleep 20
    done
  done
done
echo "all shards done"
