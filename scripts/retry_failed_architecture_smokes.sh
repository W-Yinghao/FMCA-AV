#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
FAILED_RUNS=(
  20260807-060226_cifar100-vit_s_16-32step-smoke
  20260807-060307_cifar100-convnext_tiny-32step-smoke
)
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
: > "$SUBMITTED"

retry_with_capacity_wait() {
  local source="$1" output
  while true; do
    if output="$(python3 -m harness.cli retry --run "$source" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/retry.err")"; then
      printf '%s\n' "$output"
      return 0
    fi
    if ! grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/retry.err"; then
      cat "$FMCA_HARNESS_RUN_DIR/artifacts/retry.err" >&2
      return 1
    fi
    sleep 300
    python3 -m harness.cli status >/dev/null
  done
}

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

sleep 300
RETRIES=()
for source in "${FAILED_RUNS[@]}"; do
  run_id="$(retry_with_capacity_wait "$source")"
  RETRIES+=("$run_id")
  printf '%s\t%s\n' "$source" "$run_id" >> "$SUBMITTED"
done

sleep 300
for run_id in "${RETRIES[@]}"; do
  wait_success "$run_id"
done
