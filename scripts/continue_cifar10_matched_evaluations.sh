#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || {
  echo "This continuation must run through the Slurm harness" >&2
  exit 3
}

PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG="configs/ssl/cifar10_paper_concat_smoke.json"
CORRUPTION_ROOT="/projects/EEG-foundation-model/yinghao/FMCA-AV/robustness/cifar10-c"

TRAIN_RUNS=(
  "20260807-042119_cifar10-paper-concat-20epoch"
  "20260807-042119_cifar10-gmean-matched-head-20epoch"
  "20260807-042119_cifar10-raw-first-matched-head-20epoch"
)
NAMES=("paper-concat" "gmean-matched-head" "raw-first-matched-head")
SEEDS=("20260831" "20260832" "20260833")
OVERRIDES=(
  '{"trainer":{"max_epochs":20},"probe":{"max_epochs":20}}'
  '{"model":{"parent_aggregation":"mean","f_head_hidden_dims":[2552]},"trainer":{"max_epochs":20},"probe":{"max_epochs":20}}'
  '{"model":{"parent_aggregation":"first","f_head_hidden_dims":[2552]},"trainer":{"max_epochs":20},"probe":{"max_epochs":20}}'
)

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

declare -a ACTIVE_INDEXES=()
declare -a TRAIN_CHECKPOINTS=()
declare -a PROBE_RUNS=()
declare -a PROBE_CHECKPOINTS=()

for index in "${!TRAIN_RUNS[@]}"; do
  if wait_success "${TRAIN_RUNS[$index]}"; then
    checkpoint="$(read_json_field \
      "runs/${TRAIN_RUNS[$index]}/artifacts/train_result.json" best_checkpoint)"
    ACTIVE_INDEXES+=("$index")
    TRAIN_CHECKPOINTS[$index]="$checkpoint"
  fi
done

for index in "${ACTIVE_INDEXES[@]}"; do
  run_id="$(python3 -m harness.cli submit \
    --name "cifar10-${NAMES[$index]}-20epoch-linear-probe" \
    --gpus 1 -- env \
    "FMCA_CONFIG_OVERRIDES=${OVERRIDES[$index]}" \
    "FMCA_SEED_OVERRIDE=${SEEDS[$index]}" \
    "$PYTHON" -m fmca_av.cli linear-probe \
    --config "$CONFIG" \
    --checkpoint "${TRAIN_CHECKPOINTS[$index]}")"
  PROBE_RUNS[$index]="$run_id"
  printf 'linear-probe\t%s\t%s\n' "${NAMES[$index]}" "$run_id" >> "$SUBMITTED"
done

for index in "${ACTIVE_INDEXES[@]}"; do
  if wait_success "${PROBE_RUNS[$index]}"; then
    PROBE_CHECKPOINTS[$index]="$(read_json_field \
      "runs/${PROBE_RUNS[$index]}/artifacts/probe_result.json" probe_checkpoint)"
  fi
done

declare -a KNN_RUNS=()
for index in "${ACTIVE_INDEXES[@]}"; do
  [[ -n "${PROBE_CHECKPOINTS[$index]:-}" ]] || continue
  command="$PYTHON -m fmca_av.cli knn --config $CONFIG"
  command+=" --checkpoint ${TRAIN_CHECKPOINTS[$index]}"
  command+=' --output "$FMCA_HARNESS_RUN_DIR/artifacts/knn.json"'
  command+=" --device cuda --batch-size 256 --workers 4"
  run_id="$(python3 -m harness.cli submit \
    --name "cifar10-${NAMES[$index]}-20epoch-knn" \
    --gpus 1 -- env \
    "FMCA_CONFIG_OVERRIDES=${OVERRIDES[$index]}" \
    "FMCA_SEED_OVERRIDE=${SEEDS[$index]}" \
    bash -lc "$command")"
  KNN_RUNS[$index]="$run_id"
  printf 'knn\t%s\t%s\n' "${NAMES[$index]}" "$run_id" >> "$SUBMITTED"
done

for index in "${ACTIVE_INDEXES[@]}"; do
  [[ -n "${KNN_RUNS[$index]:-}" ]] && wait_success "${KNN_RUNS[$index]}" || true
done

declare -a CORRUPTION_RUNS=()
for index in "${ACTIVE_INDEXES[@]}"; do
  [[ -n "${PROBE_CHECKPOINTS[$index]:-}" ]] || continue
  command="$PYTHON -m fmca_av.cli corruption-eval --config $CONFIG"
  command+=" --checkpoint ${TRAIN_CHECKPOINTS[$index]}"
  command+=" --probe-checkpoint ${PROBE_CHECKPOINTS[$index]}"
  command+=" --root $CORRUPTION_ROOT"
  command+=' --output "$FMCA_HARNESS_RUN_DIR/artifacts/cifar10c.json"'
  command+=" --device cuda --batch-size 256 --workers 0"
  run_id="$(python3 -m harness.cli submit \
    --name "cifar10-${NAMES[$index]}-20epoch-cifar10c" \
    --gpus 1 -- env \
    "FMCA_CONFIG_OVERRIDES=${OVERRIDES[$index]}" \
    "FMCA_SEED_OVERRIDE=${SEEDS[$index]}" \
    bash -lc "$command")"
  CORRUPTION_RUNS[$index]="$run_id"
  printf 'cifar10-c\t%s\t%s\n' "${NAMES[$index]}" "$run_id" >> "$SUBMITTED"
done

for index in "${ACTIVE_INDEXES[@]}"; do
  [[ -n "${CORRUPTION_RUNS[$index]:-}" ]] && wait_success "${CORRUPTION_RUNS[$index]}" || true
done

python3 -m harness.cli submit --name summarize-matched-cifar10-experiments --gpus 0 \
  -- "$PYTHON" scripts/summarize_results.py
