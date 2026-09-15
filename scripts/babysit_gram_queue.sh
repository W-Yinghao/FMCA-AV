#!/bin/bash
# Feed the remaining validity-gate jobs in as submit slots free up.
# The submit cap is per user across partitions, so CPU work still has to
# wait for a slot even though it runs on its own nodes.
cd /home/infres/yinwang/FMCA-AV
PY=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
PENDING=scripts/gram_pending.txt
while [ -s "$PENDING" ]; do
  LINE=$(head -1 "$PENDING")
  NAME=$(echo "$LINE" | cut -d'|' -f1)
  ARGS=$(echo "$LINE" | cut -d'|' -f2-)
  OUT=$(echo "$ARGS" | sed 's/.*--out //' | awk '{print $1}')
  if [ -f "$OUT" ]; then
    sed -i 1d "$PENDING"; continue
  fi
  if sbatch --job-name="$NAME" --partition=CPU --cpus-per-task=16 --mem=48G \
       --time=8:00:00 --output="runs/${NAME}_%j.out" --wrap="$PY $ARGS" >/dev/null 2>&1; then
    echo "submitted $NAME ($(wc -l < "$PENDING") left after this)"
    sed -i 1d "$PENDING"
  else
    sleep 300
  fi
done
echo "VALIDITY QUEUE DRAINED"
