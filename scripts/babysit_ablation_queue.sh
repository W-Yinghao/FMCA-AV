#!/bin/bash
# Pops pending sbatch lines whenever the user's queue has headroom.
# Loud failures: a line that fails to submit goes to the .failed file
# with the error, and the babysitter moves on.
PENDING=scripts/ablation_pending.txt
LIMIT=28
cd /home/infres/yinwang/FMCA-AV
while [ -s "$PENDING" ]; do
  count=$(squeue -u "$USER" -h | wc -l)
  while [ "$count" -lt "$LIMIT" ] && [ -s "$PENDING" ]; do
    line=$(head -1 "$PENDING")
    if out=$($line 2>&1); then
      echo "$(date '+%F %T') OK   $line -> $out" >> runs/babysit_ablation.log
    else
      echo "$(date '+%F %T') FAIL $line -> $out" >> "$PENDING.failed"
    fi
    tail -n +2 "$PENDING" > "$PENDING.tmp" && mv "$PENDING.tmp" "$PENDING"
    count=$((count + 1))
  done
  sleep 600
done
echo "$(date '+%F %T') pending list drained" >> runs/babysit_ablation.log
