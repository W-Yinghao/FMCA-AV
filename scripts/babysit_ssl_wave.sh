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

# Single instance, enforced by an atomic mkdir rather than by inspecting the
# process table: two babysitters double-submit, and a dead one silently
# releases nothing.  Whichever outcome you get here is definitive.
LOCK=runs/.babysit_ssl.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(date '+%F %T') another babysitter holds $LOCK; exiting" >> runs/babysit_ssl.log
  exit 3
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PENDING=scripts/ssl_wave_pending.txt
LIMIT=29
PY=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python

probe_status () {
  "$PY" -c "
import json
try: print(json.load(open('$1')).get('status','missing'))
except Exception: print('missing')"
}

# Run until told to stop, not until the list is momentarily empty: work is
# appended to this list while the babysitter runs, and an exit-on-drained
# loop silently stops watching exactly when more arrives.  That happened --
# 17 lines sat unsubmitted on an idle partition.
STOP=runs/.babysit_ssl.stop
rm -f "$STOP"
while [ ! -f "$STOP" ]; do
  count=$(squeue -u "$USER" -h | wc -l)
  rotated=0
  total=$(wc -l < "$PENDING")
  while [ "$count" -lt "$LIMIT" ] && [ -s "$PENDING" ] && [ "$rotated" -lt "$total" ]; do
    line=$(head -1 "$PENDING")
    tail -n +2 "$PENDING" > "$PENDING.tmp" && mv "$PENDING.tmp" "$PENDING"
    case "$line" in
      GATED:*)
        # GATED:<probe-unit.json>:<command> -- each arm waits on ITS OWN
        # probe, so one arm's failure cannot release or block another's.
        rest=${line#GATED:}
        probe=${rest%%:*}
        status=$(probe_status "$probe")
        if [ "$status" != "complete" ]; then
          echo "$line" >> "$PENDING"          # back of the queue, try later
          rotated=$((rotated + 1))
          continue
        fi
        line=${rest#*:}
        ;;
    esac
    # Idempotent submission.  A babysitter instance started before the lock
    # existed is not bound by it, and two instances releasing the same gated
    # line would put two jobs into one unit directory.  The ledger makes a
    # repeat submission a no-op no matter how many instances there are.
    LEDGER=runs/babysit_ssl.submitted
    touch "$LEDGER"
    if grep -Fxq "$line" "$LEDGER"; then
      echo "$(date '+%F %T') SKIP already submitted: $line" >> runs/babysit_ssl.log
      continue
    fi
    echo "$line" >> "$LEDGER"
    if out=$(eval "$line" 2>&1); then
      echo "$(date '+%F %T') OK   $line -> $out" >> runs/babysit_ssl.log
    else
      echo "$(date '+%F %T') FAIL $line -> $out" >> "$PENDING.failed"
    fi
    count=$((count + 1))
  done
  # Heartbeat: gated lines rotate without logging, so without this a live
  # babysitter and a dead one look identical in the log.
  echo "$(date '+%F %T') alive: $(wc -l < "$PENDING") pending, $(squeue -u "$USER" -h | wc -l) queued" \
    >> runs/babysit_ssl.log
  sleep 420
done
echo "$(date '+%F %T') stopped by $STOP" >> runs/babysit_ssl.log
