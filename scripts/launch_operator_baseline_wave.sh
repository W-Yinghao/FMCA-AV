#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
: > "$SUBMITTED"

state_of() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$1/status.json"; }
submit_with_capacity_wait() {
  local output
  while true; do
    if output="$(python3 -m harness.cli submit "$@" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then printf '%s\n' "$output"; return 0; fi
    if ! grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err"; then cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2; return 1; fi
    sleep 300; python3 -m harness.cli status >/dev/null
  done
}
wait_success() {
  local run_id="$1" state
  while true; do
    python3 -m harness.cli status --run "$run_id" >/dev/null; state="$(state_of "$run_id")"
    case "$state" in SUCCEEDED) return 0;; FAILED|STOPPED|BLOCKED) echo "$run_id ended in $state" >&2; return 1;; *) sleep 300;; esac
  done
}

sleep 300
# The already completed 9-view concat run is retained as the explicitly marked
# HFMCA-style reimplementation; no unavailable original checkpoint is implied.
printf 'hfmca_style_reimplementation\t%s\n' 20260807-040405_cifar10-paper-9view-concat-5epoch-smoke >> "$SUBMITTED"
printf 'fmca_av_m2\t%s\n' 20260807-035405_cifar10-lightning-5epoch-smoke >> "$SUBMITTED"

override='{"experiment":{"name":"cifar10-regular-fmca-m1"},"data":{"num_views":1,"include_raw_parent":true},"model":{"parent_aggregation":"raw"}}'
REGULAR="$(submit_with_capacity_wait --name cifar10-regular-fmca-m1-5epoch --gpus 1 -- env "FMCA_CONFIG_OVERRIDES=$override" bash scripts/run_fmca_pipeline.sh --config configs/ssl/cifar10_smoke.json)"
printf 'regular_fmca_m1\t%s\n' "$REGULAR" >> "$SUBMITTED"

METHODS=(dcca vamp2)
TRAIN_RUNS=()
for index in "${!METHODS[@]}"; do
  method="${METHODS[$index]}"
  override="{\"experiment\":{\"method\":\"$method\"},\"objective\":{\"ridge\":0.001}}"
  run_id="$(submit_with_capacity_wait --name "cifar10-${method}-5epoch-screening" --gpus 1 -- env \
    "FMCA_CONFIG_OVERRIDES=$override" "FMCA_SEED_OVERRIDE=$((20260870 + index))" \
    "$PYTHON" -m fmca_av.baseline_cli train --config configs/ssl/cifar10_baseline_smoke.json)"
  TRAIN_RUNS+=("$run_id"); printf '%s\t%s\n' "$method" "$run_id" >> "$SUBMITTED"
done

sleep 300
wait_success "$REGULAR"
checkpoint="$($PYTHON -c 'import json,sys; result=json.load(open(sys.argv[1])); print(result.get("best_checkpoint") or result.get("last_checkpoint") or "")' "runs/$REGULAR/artifacts/train_result.json")"
run_id="$(submit_with_capacity_wait --name cifar10-regular-fmca-m1-linear-probe --gpus 1 -- env \
  'FMCA_CONFIG_OVERRIDES={"data":{"num_views":1,"include_raw_parent":true},"model":{"parent_aggregation":"raw"}}' \
  "$PYTHON" -m fmca_av.cli linear-probe --config configs/ssl/cifar10_smoke.json --checkpoint "$checkpoint")"
printf 'regular_fmca_probe\t%s\n' "$run_id" >> "$SUBMITTED"

for index in "${!METHODS[@]}"; do
  method="${METHODS[$index]}"; train_run="${TRAIN_RUNS[$index]}"; wait_success "$train_run"
  checkpoint="$($PYTHON -c 'import json,sys; result=json.load(open(sys.argv[1])); print(result.get("best_checkpoint") or result.get("last_checkpoint") or "")' "runs/$train_run/artifacts/train_result.json")"
  override="{\"experiment\":{\"method\":\"$method\"},\"objective\":{\"ridge\":0.001}}"
  run_id="$(submit_with_capacity_wait --name "cifar10-${method}-5epoch-linear-probe" --gpus 1 -- env \
    "FMCA_CONFIG_OVERRIDES=$override" "$PYTHON" -m fmca_av.baseline_cli linear-probe \
    --config configs/ssl/cifar10_baseline_smoke.json --checkpoint "$checkpoint")"
  printf '%s_probe\t%s\n' "$method" "$run_id" >> "$SUBMITTED"
done
