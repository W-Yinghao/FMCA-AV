#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || {
  echo "This wave launcher must run through the Slurm harness" >&2
  exit 3
}

PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
: > "$SUBMITTED"

active_gpus() {
  python3 -m harness.cli status >/dev/null
  "$PYTHON" - <<'PY'
import json
x=json.load(open('harness/state/jobs.json'))['jobs'].values()
print(sum(int(job.get('requested_gpus',0)) for job in x if job.get('state') in {'QUEUED','RUNNING'}))
PY
}

wait_for_slot() {
  while [[ "$(active_gpus)" -ge 4 ]]; do sleep 300; done
}

wait_for_slot
run_id="$(python3 -m harness.cli submit --name imagenet100-lightning-32step-smoke \
  --gpus 1 --profile imagenet -- bash scripts/run_fmca_pipeline.sh \
  --config configs/ssl/imagenet100_smoke.json)"
printf 'imagenet100\t%s\n' "$run_id" >> "$SUBMITTED"

wait_for_slot
run_id="$(python3 -m harness.cli submit --name stl10-lightning-5epoch-smoke \
  --gpus 1 -- bash scripts/run_fmca_pipeline.sh --config configs/ssl/stl10_smoke.json)"
printf 'stl10\t%s\n' "$run_id" >> "$SUBMITTED"

wait_for_slot
run_id="$(python3 -m harness.cli submit --name tinyimagenet200-lightning-5epoch-smoke \
  --gpus 1 -- bash scripts/run_fmca_pipeline.sh --config configs/ssl/tinyimagenet200_smoke.json)"
printf 'tinyimagenet200\t%s\n' "$run_id" >> "$SUBMITTED"

wait_for_slot
run_id="$(python3 -m harness.cli submit --name cifar10-simclr-5epoch-screening \
  --gpus 1 -- env FMCA_SEED_OVERRIDE=20260850 \
  "$PYTHON" -m fmca_av.baseline_cli train --config configs/ssl/cifar10_baseline_smoke.json)"
printf 'simclr\t%s\n' "$run_id" >> "$SUBMITTED"
