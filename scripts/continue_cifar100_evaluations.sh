#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || {
  echo "This continuation must run through the Slurm harness" >&2
  exit 3
}

PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG="configs/ssl/cifar100_smoke.json"
TRAIN_RUN="20260807-043435_cifar100-lightning-20epoch"
OVERRIDES='{"trainer":{"max_epochs":20},"probe":{"max_epochs":20}}'
SEED="20260921"
CORRUPTION_ROOT="/projects/EEG-foundation-model/yinghao/FMCA-AV/robustness/cifar100-c"

mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
: > "$SUBMITTED"

job_state() {
  local run_id="$1"
  python3 -m harness.cli status --run "$run_id" >/dev/null
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' \
    "runs/$run_id/status.json"
}

wait_success() {
  local run_id="$1"
  local state
  while true; do
    state="$(job_state "$run_id")"
    case "$state" in
      SUCCEEDED) return 0 ;;
      FAILED|STOPPED|BLOCKED)
        echo "$run_id ended in $state" >&2
        return 1
        ;;
      *) sleep 300 ;;
    esac
  done
}

read_json_field() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

wait_success "$TRAIN_RUN"
TRAIN_CHECKPOINT="$(read_json_field \
  "runs/$TRAIN_RUN/artifacts/train_result.json" best_checkpoint)"

PROBE_RUN="$(python3 -m harness.cli submit \
  --name cifar100-20epoch-linear-probe --gpus 1 -- env \
  "FMCA_CONFIG_OVERRIDES=$OVERRIDES" "FMCA_SEED_OVERRIDE=$SEED" \
  "$PYTHON" -m fmca_av.cli linear-probe \
  --config "$CONFIG" --checkpoint "$TRAIN_CHECKPOINT")"
printf 'linear-probe\t%s\n' "$PROBE_RUN" >> "$SUBMITTED"
wait_success "$PROBE_RUN"
PROBE_CHECKPOINT="$(read_json_field \
  "runs/$PROBE_RUN/artifacts/probe_result.json" probe_checkpoint)"

command="$PYTHON -m fmca_av.cli knn --config $CONFIG"
command+=" --checkpoint $TRAIN_CHECKPOINT"
command+=' --output "$FMCA_HARNESS_RUN_DIR/artifacts/knn.json"'
command+=" --device cuda --batch-size 256 --workers 4"
KNN_RUN="$(python3 -m harness.cli submit \
  --name cifar100-20epoch-knn --gpus 1 -- env \
  "FMCA_CONFIG_OVERRIDES=$OVERRIDES" "FMCA_SEED_OVERRIDE=$SEED" \
  bash -lc "$command")"
printf 'knn\t%s\n' "$KNN_RUN" >> "$SUBMITTED"
wait_success "$KNN_RUN"

command="$PYTHON -m fmca_av.cli corruption-eval --config $CONFIG"
command+=" --checkpoint $TRAIN_CHECKPOINT --probe-checkpoint $PROBE_CHECKPOINT"
command+=" --root $CORRUPTION_ROOT"
command+=' --output "$FMCA_HARNESS_RUN_DIR/artifacts/cifar100c.json"'
command+=" --device cuda --batch-size 256 --workers 0"
CORRUPTION_RUN="$(python3 -m harness.cli submit \
  --name cifar100-20epoch-cifar100c --gpus 1 -- env \
  "FMCA_CONFIG_OVERRIDES=$OVERRIDES" "FMCA_SEED_OVERRIDE=$SEED" \
  bash -lc "$command")"
printf 'cifar100-c\t%s\n' "$CORRUPTION_RUN" >> "$SUBMITTED"
wait_success "$CORRUPTION_RUN"

python3 -m harness.cli submit --name summarize-cifar100-experiments --gpus 0 \
  -- "$PYTHON" scripts/summarize_results.py
