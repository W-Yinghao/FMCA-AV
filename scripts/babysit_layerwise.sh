#!/bin/bash
# Wait for the smoke unit; if its profile is valid, release the fleet.
cd /home/infres/yinwang/FMCA-AV
PY=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
while squeue -j "$1" -h 2>/dev/null | grep -q .; do sleep 60; done
PROFILE=results/gate1/gate1_20260820_v8/units/product_endpoint__seed1/layerwise_profile.json
if ! $PY -c "
import json,sys
d=json.load(open('$PROFILE'))
assert d['status']=='complete' and len(d['stages'])==4, 'bad profile'
print('SMOKE OK  probe by stage:', [round(s['probe']['test_accuracy']*100,2) for s in d['stages']])
print('          eff rank     :', [round(s['spectrum']['effective_rank'],1) for s in d['stages']])
" 2>&1; then
  echo "SMOKE FAILED - fleet not released"; tail -20 runs/lwsmoke_$1.out; exit 1
fi
submit () {  # config-dir output-root variant seed
  sbatch --job-name=lwprof --partition=A100,H100,L40S --exclude=node51 --gres=gpu:1 \
    --cpus-per-task=8 --mem=48G --output=runs/lwprof_%j.out \
    --wrap="$PY scripts/run_layerwise_profile.py --config-dir $1 --output-root $2 --variant $3 --seed $4" >/dev/null
}
N=0
for seed in 1 2 3; do
  for variant in product_endpoint final_mview additive_mview; do
    [ "$variant$seed" = "product_endpoint1" ] && continue   # the smoke unit
    submit configs/gate_v8 results/gate1/gate1_20260820_v8 $variant $seed; N=$((N+1))
    submit configs/gate_c100 results/gate1/gate1_20260821_c100pilot $variant $seed; N=$((N+1))
  done
done
submit configs/gate_c100 results/gate1/gate1_20260821_c100pilot product_endpoint 1; N=$((N+1))
submit configs/gate_v8_alpha0 results/gate1/gate1_20260821_v8_alpha0 product_endpoint 1; N=$((N+1))
echo "FLEET RELEASED: $N profile jobs"
