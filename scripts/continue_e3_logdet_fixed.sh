#!/usr/bin/env bash
set -euo pipefail
[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
sleep 300
while true; do
  if run_id="$(python3 -m harness.cli submit --name e3-cifar10-objective-logdet-fp64-fixed --gpus 1 -- \
    env 'FMCA_CONFIG_OVERRIDES={"objective":{"name":"logdet"}}' FMCA_SEED_OVERRIDE=20269001 \
    bash scripts/run_fmca_pipeline.sh --config configs/ssl/cifar10_smoke.json \
    2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then
    printf '%s\n' "$run_id" > "$FMCA_HARNESS_RUN_DIR/artifacts/submitted_run_id.txt"
    exit 0
  fi
  grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" || { cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2; exit 1; }
  sleep 300; python3 -m harness.cli status >/dev/null
done
