#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
TORCHRUN="/home/infres/yinwang/FMCA-AV/scripts/torchrun"
TSD_WATCHER="20260807-060445_launch-cifar10-tsd-full-severity-sweep-fixed"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
: > "$SUBMITTED"

wait_success() {
  local run_id="$1" state
  while true; do
    python3 -m harness.cli status --run "$run_id" >/dev/null
    state="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$run_id/status.json")"
    case "$state" in
      SUCCEEDED) return 0 ;;
      FAILED|STOPPED|BLOCKED) echo "$run_id ended in $state" >&2; return 1 ;;
      *) sleep 300 ;;
    esac
  done
}

submit_with_capacity_wait() {
  local output
  while true; do
    if output="$(python3 -m harness.cli submit "$@" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then printf '%s\n' "$output"; return 0; fi
    if ! grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err"; then cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2; return 1; fi
    sleep 300; python3 -m harness.cli status >/dev/null
  done
}

sleep 300
wait_success "$TSD_WATCHER"

run_id="$(submit_with_capacity_wait --name e10-cifar10-ddp1-100step-scaling --gpus 1 --profile v100 -- \
  bash scripts/run_fmca_pipeline.sh --config configs/ssl/cifar10_ddp1_smoke.json)"
printf '1\t%s\n' "$run_id" >> "$SUBMITTED"
sleep 300; wait_success "$run_id"

run_id="$(submit_with_capacity_wait --name e10-cifar10-ddp2-100step-scaling --gpus 2 --profile v100 -- \
  "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=2 -m scripts.run_fmca_pipeline \
  --config configs/ssl/cifar10_ddp2_smoke.json)"
printf '2\t%s\n' "$run_id" >> "$SUBMITTED"
sleep 300; wait_success "$run_id"
