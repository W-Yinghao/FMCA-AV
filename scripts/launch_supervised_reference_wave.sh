#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"; : > "$SUBMITTED"

submit_with_capacity_wait() {
  local output
  while true; do
    if output="$(python3 -m harness.cli submit "$@" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then printf '%s\n' "$output"; return 0; fi
    if ! grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err"; then cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2; return 1; fi
    sleep 300; python3 -m harness.cli status >/dev/null
  done
}

sleep 300
run_id="$(submit_with_capacity_wait --name supervised-cifar10-reference-5epoch --gpus 1 -- \
  "$PYTHON" -m fmca_av.supervised_cli --config configs/ssl/cifar10_baseline_smoke.json)"
printf 'cifar10\t%s\n' "$run_id" >> "$SUBMITTED"

run_id="$(submit_with_capacity_wait --name supervised-imagenet100-reference-32step --gpus 1 --profile imagenet -- \
  "$PYTHON" -m fmca_av.supervised_cli --config configs/ssl/imagenet100_smoke.json)"
printf 'imagenet100\t%s\n' "$run_id" >> "$SUBMITTED"

run_id="$(submit_with_capacity_wait --name supervised-imagenet1k-reference-32step --gpus 1 --profile imagenet -- \
  "$PYTHON" -m fmca_av.supervised_cli --config configs/ssl/imagenet1k_smoke.json)"
printf 'imagenet1k\t%s\n' "$run_id" >> "$SUBMITTED"
