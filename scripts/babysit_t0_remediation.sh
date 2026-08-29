#!/bin/bash
# Gate the enlarged-calibration retraining on the CPU diagnostic.
# Frozen rule (T0_REMEDIATION_PREREG_FROZEN_20260824.md): if the
# hierarchical arms still move more than 0.02 over the last doubling at
# N=10000, then n_calibration = n_val = 10000 is not the right target
# either and the retraining is re-planned, not launched.
cd /home/infres/yinwang/FMCA-AV
PY=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
while squeue -u yinwang -h -o "%j" | grep -q vg_ext; do sleep 300; done

VERDICT=$($PY - <<'PYEOF'
import json, os
worst, seen = 0.0, 0
for tag in ("c10", "c100"):
    for v in ("product_endpoint", "additive_mview"):
        p = f"results/validity/{tag}_convext_{v}.json"
        if not os.path.isfile(p):
            continue
        curve = json.load(open(p))["curve"]
        keys = sorted(curve, key=int)
        if len(keys) < 2:
            continue
        change = abs(curve[keys[-1]]["normalized_closure_defect"]
                     - curve[keys[-2]]["normalized_closure_defect"])
        print(f"# {tag} {v}: N={keys[-2]}->{keys[-1]} change={change:.4f} "
              f"final={curve[keys[-1]]['normalized_closure_defect']:.4f}")
        worst = max(worst, change); seen += 1
print("PROCEED" if seen and worst < 0.02 else "REPLAN", f"worst={worst:.4f}", f"arms={seen}")
PYEOF
)
echo "$VERDICT"
if ! echo "$VERDICT" | tail -1 | grep -q PROCEED; then
  echo "T0 REMEDIATION: budget re-plan needed, retraining NOT launched"
  exit 0
fi

echo "T0 REMEDIATION: diagnostic passed, feeding the retraining fleet"
for seed in 1 2 3; do
  for variant in product_endpoint additive_mview final_mview; do
    for spec in "configs/gate_v8_bigcal results/gate1/gate1_20260824_v8_bigcal" \
                "configs/gate_c100_bigcal results/gate1/gate1_20260824_c100_bigcal"; do
      set -- $spec
      until sbatch scripts/launch_gate1_generic.sbatch "$1" "$2" "$variant" "$seed" >/dev/null 2>&1; do
        sleep 300
      done
      echo "submitted $variant seed$seed -> $2"
    done
  done
done
echo "T0 RETRAINING FLEET RELEASED: 18 units"
