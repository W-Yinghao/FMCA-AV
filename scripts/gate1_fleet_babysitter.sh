#!/bin/bash
# Gate 1 slot-watcher: submits remaining fleet units as QOS headroom frees.
# Aborts if the QC probe unit did not complete.
set -u
cd /home/infres/yinwang/FMCA-AV
PROBE_JOB=945112
PROBE_RECORD=results/gate1/gate1_20260816_v1/probe/product_endpoint__seed1/unit.json
UNITS="additive_mview:1 additive_mview:2 additive_mview:3 amdim_cross:1 amdim_cross:2 amdim_cross:3 product_only:1 product_only:2 product_only:3 product_endpoint:1 product_endpoint:2 product_endpoint:3"

for unit in $UNITS; do
  variant=${unit%:*}; seed=${unit#*:}
  while true; do
    count=$(squeue -u yinwang -h | wc -l)
    if [ "$count" -ge 22 ]; then sleep 120; continue; fi
    if squeue -j $PROBE_JOB -h 2>/dev/null | grep -q .; then
      DEP="--dependency=afterok:$PROBE_JOB"
    else
      status=$(/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python -c "import json;print(json.load(open('$PROBE_RECORD')).get('status'))" 2>/dev/null || echo missing)
      if [ "$status" != "complete" ]; then
        echo "PROBE_GATE_FAILED status=$status - aborting fleet submission at $unit"
        exit 1
      fi
      DEP=""
    fi
    if sbatch $DEP scripts/launch_gate1_unit.sbatch "$variant" "$seed"; then
      echo "submitted $unit"
      break
    fi
    sleep 120
  done
done
echo "ALL_FLEET_UNITS_SUBMITTED"
