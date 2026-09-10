#!/bin/bash
# Slot-watcher for the five-question SSL wave.
#
# Lines whose runner has a passing QC probe are submitted freely.  Lines
# tagged GATED: are held until the named probe artefact reports
# "complete" -- a refilling babysitter in front of an unproven runner is
# how 15h once disappeared, and the type-2 arm has already failed two
# probes.  A gated line that is not yet cleared is rotated to the BACK of
# the list, so it never blocks the rest of the wave.
set -u
cd /home/infres/yinwang/FMCA-AV
PENDING=scripts/ssl_wave_pending.txt
GRAM_PROBE=results/gate1/gate1_20260910_gram_probe/units/product_endpoint__seed1/unit.json
LIMIT=29
PY=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python

probe_status () {
  "$PY" -c "
import json
try: print(json.load(open('$1')).get('status','missing'))
except Exception: print('missing')"
}

while [ -s "$PENDING" ]; do
  count=$(squeue -u "$USER" -h | wc -l)
  rotated=0
  total=$(wc -l < "$PENDING")
  while [ "$count" -lt "$LIMIT" ] && [ -s "$PENDING" ] && [ "$rotated" -lt "$total" ]; do
    line=$(head -1 "$PENDING")
    tail -n +2 "$PENDING" > "$PENDING.tmp" && mv "$PENDING.tmp" "$PENDING"
    case "$line" in
      GATED:*)
        status=$(probe_status "$GRAM_PROBE")
        if [ "$status" != "complete" ]; then
          echo "$line" >> "$PENDING"          # back of the queue, try later
          rotated=$((rotated + 1))
          continue
        fi
        line=${line#GATED:}
        ;;
    esac
    if out=$(eval "$line" 2>&1); then
      echo "$(date '+%F %T') OK   $line -> $out" >> runs/babysit_ssl.log
    else
      echo "$(date '+%F %T') FAIL $line -> $out" >> "$PENDING.failed"
    fi
    count=$((count + 1))
  done
  sleep 420
done
echo "$(date '+%F %T') ssl wave pending drained" >> runs/babysit_ssl.log
