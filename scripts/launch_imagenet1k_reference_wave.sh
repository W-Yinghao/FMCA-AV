#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
TORCHRUN="/home/infres/yinwang/FMCA-AV/scripts/torchrun"
TSD_WATCHER="20260807-060445_launch-cifar10-tsd-full-severity-sweep-fixed"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
: > "$SUBMITTED"

state_of() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$1/status.json"; }
wait_success() {
  local run_id="$1" state
  while true; do
    python3 -m harness.cli status --run "$run_id" >/dev/null; state="$(state_of "$run_id")"
    case "$state" in SUCCEEDED) return 0;; FAILED|STOPPED|BLOCKED) echo "$run_id ended in $state" >&2; return 1;; *) sleep 300;; esac
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
SEEDS=(20267001 20267002 20267003)
TRAIN_RUNS=()
for seed in "${SEEDS[@]}"; do
  run_id="$(submit_with_capacity_wait --name "imagenet1k-fmca-av-reference-seed-${seed}" --gpus 2 --profile imagenet_ddp -- env \
    "FMCA_SEED_OVERRIDE=$seed" "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=2 \
    -m scripts.run_fmca_pipeline --config configs/ssl/imagenet1k_reference.json)"
  TRAIN_RUNS+=("$run_id"); printf 'train\t%s\t%s\n' "$seed" "$run_id" >> "$SUBMITTED"
  sleep 300; wait_success "$run_id"
done

for index in "${!TRAIN_RUNS[@]}"; do
  seed="${SEEDS[$index]}"; train_run="${TRAIN_RUNS[$index]}"
  checkpoint="$($PYTHON -c 'import json,sys; result=json.load(open(sys.argv[1])); print(result.get("best_checkpoint") or result.get("last_checkpoint") or "")' "runs/$train_run/artifacts/train_result.json")"
  override='{"probe":{"devices":1,"max_epochs":90,"accelerator":"gpu"}}'
  run_id="$(submit_with_capacity_wait --name "imagenet1k-reference-linear-probe-seed-${seed}" --gpus 1 --profile imagenet -- env \
    "FMCA_CONFIG_OVERRIDES=$override" "FMCA_SEED_OVERRIDE=$seed" "$PYTHON" -m fmca_av.cli linear-probe \
    --config configs/ssl/imagenet1k_reference.json --checkpoint "$checkpoint")"
  printf 'probe\t%s\t%s\n' "$seed" "$run_id" >> "$SUBMITTED"
done
