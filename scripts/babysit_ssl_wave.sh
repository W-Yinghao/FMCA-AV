#!/bin/bash
# Slot-watcher for the five-question SSL wave.
#
# GATED: this refuses to start unless the QC probe artefacts it depends
# on exist and passed.  A refilling babysitter in front of a broken
# runner is exactly how 15h once disappeared.
set -u
cd /home/infres/yinwang/FMCA-AV
PENDING=scripts/ssl_wave_pending.txt
GATE_PROBE=results/gate1/gate1_20260910_gram_probe/units/product_endpoint__seed1/unit.json
LIMIT=28

status=$(/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python -c "
import json,sys
try: print(json.load(open('$GATE_PROBE')).get('status','missing'))
except Exception: print('missing')")
if [ "$status" != "complete" ]; then
  echo "$(date '+%F %T') REFUSED to start: gram probe status=$status" >> runs/babysit_ssl.log
  exit 1
fi

while [ -s "$PENDING" ]; do
  count=$(squeue -u "$USER" -h | wc -l)
  while [ "$count" -lt "$LIMIT" ] && [ -s "$PENDING" ]; do
    line=$(head -1 "$PENDING")
    if out=$($line 2>&1); then
      echo "$(date '+%F %T') OK   $line -> $out" >> runs/babysit_ssl.log
    else
      echo "$(date '+%F %T') FAIL $line -> $out" >> "$PENDING.failed"
    fi
    tail -n +2 "$PENDING" > "$PENDING.tmp" && mv "$PENDING.tmp" "$PENDING"
    count=$((count + 1))
  done
  sleep 600
done
echo "$(date '+%F %T') ssl wave pending drained" >> runs/babysit_ssl.log
