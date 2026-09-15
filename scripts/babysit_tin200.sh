#!/bin/bash
# Probe-gate the Tiny ImageNet fleet: release only if the QC probe survives.
cd /home/infres/yinwang/FMCA-AV
PY=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
while squeue -j "$1" -h 2>/dev/null | grep -q .; do sleep 60; done
REC=$(ls results/gate1/gate1_20260823_tin200*/probe/product_endpoint__seed1/unit.json 2>/dev/null | head -1)
if [ -z "$REC" ] || ! $PY -c "
import json,sys
d=json.load(open('$REC'))
assert d['status']=='complete', d.get('reason','no status')
print('QC PROBE OK  probe=%.2f%%  defect=%.3f  wall=%.0fs' % (
    d['linear_probe']['test_accuracy']*100, d['certificate']['normalized_closure_defect'], d['wall_seconds']))
" 2>&1; then
  echo "QC PROBE FAILED - fleet not released"
  tail -25 runs/*"$1"*.out 2>/dev/null
  exit 1
fi
for variant in product_endpoint final_mview additive_mview; do
  sbatch scripts/launch_gate1_generic.sbatch configs/gate_tin results/gate1/gate1_20260823_tin200 $variant 1 >/dev/null
done
echo "TIN200 FLEET RELEASED: 3 units (one seed, V7 / flat / additive)"
