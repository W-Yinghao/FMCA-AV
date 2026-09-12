#!/bin/bash
# Slot-filling submitter driven by a target-state sweep.
#
# The difference from a static pending list: the work to do is recomputed
# from disk every cycle, so a unit that lands at 04:00 has its certificate
# and its profile queued at 04:05 without anyone editing a file.  The
# previous list-based babysitter left a tail behind every time the fleet
# outran the last queueing pass.
set -u
cd /home/infres/yinwang/FMCA-AV

LOCK=runs/.babysit_sweep.lock
LOG=runs/babysit_sweep.log
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(date '+%F %T') another sweep babysitter holds $LOCK; exiting" >> "$LOG"
  exit 3
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

PY=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
LIMIT=29                      # leave slot 30 free for a hand-submitted probe
STOP=runs/.babysit_sweep.stop
rm -f "$STOP"

while [ ! -f "$STOP" ]; do
  count=$(squeue -u "$USER" -h | wc -l)
  submitted=0
  if [ "$count" -lt "$LIMIT" ]; then
    # Recomputed here, inside the loop, on purpose.
    "$PY" scripts/sweep_pending.py --with-keys > runs/.sweep.tmp 2>> "$LOG"
    while [ "$count" -lt "$LIMIT" ] && IFS=$'\t' read -r key cmd; do
      [ -z "${key:-}" ] && continue
      if out=$(eval "$cmd" 2>&1); then
        job=$(printf '%s' "$out" | grep -oE '[0-9]+$')
        # Recorded BEFORE the next sweep can run, so a slow sbatch cannot
        # produce a duplicate.  inflight is pruned against squeue; attempts
        # is never pruned and is what stops a crash-loop.
        printf '%s\t%s\n' "$key" "$job" >> runs/inflight.tsv
        printf '%s\t%s\n' "$key" "$job" >> runs/attempts.tsv
        echo "$(date '+%F %T') OK   $key -> $job" >> "$LOG"
        count=$((count + 1))
        submitted=$((submitted + 1))
      else
        echo "$(date '+%F %T') FAIL $key -> $out" >> "$LOG"
      fi
    done < runs/.sweep.tmp
  fi
  pending=$("$PY" scripts/sweep_pending.py 2>/dev/null | wc -l)
  blocked=$("$PY" scripts/sweep_pending.py --blocked 2>/dev/null | wc -l)
  echo "$(date '+%F %T') alive: submitted $submitted, $pending still to do, $blocked blocked, $(squeue -u "$USER" -h | wc -l) queued" >> "$LOG"
  sleep 300
done
echo "$(date '+%F %T') stopped by $STOP" >> "$LOG"
