#!/bin/bash
# Night watchdog: a job whose log could not be created (intermittent "Disk quota exceeded") is moved to done/ without
# running. Detect that from 15902410.err and put the job back into the queue (max 5 retries per job). Our jobs only.
cd /p/scratch/training2666/neves2/BlitzTeam/Hackathon_Watermark_Localisation_Task/benelhajyahia1
Q=slurm/queue; seen=$(wc -l < output/15902410.err)
while true; do
  n=$(wc -l < output/15902410.err)
  if [ "$n" -gt "$seen" ]; then
    tail -n +$((seen + 1)) output/15902410.err | grep -o "output/15902410_[0-9]*_[A-Za-z0-9_]*\.log: Disk quota exceeded" | while read l; do
      b=$(echo "$l" | sed 's|output/15902410_||; s|\.log:.*||'); f=$Q/done/$b.cmd
      if [ -f "$f" ] && [ "$(stat -c %U $f)" = "benelhajyahia1" ]; then
        r=$(grep -c "$b" /tmp/claude-37358/requeued.txt 2>/dev/null || echo 0)
        if [ "$r" -lt 5 ]; then
          new=$Q/$(date +%s%N)_${b#*_}.cmd
          for t in 1 2 3 4 5 6; do cp "$f" "$new" 2>/dev/null && break; sleep 10; done
          echo "$(date +%H:%M:%S) requeued $b -> $(basename $new)" | tee -a /tmp/claude-37358/requeued.txt
        fi
      fi
    done
    seen=$n
  fi
  sleep 30
done
