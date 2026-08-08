#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG="configs/ssl/cifar10_paper_concat_smoke.json"
ROOT="/projects/EEG-foundation-model/yinghao/FMCA-AV/robustness/cifar10-c"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
: > "$SUBMITTED"

submit_with_capacity_wait() {
  local output
  while true; do
    if output="$(python3 -m harness.cli submit "$@" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then
      printf '%s\n' "$output"
      return 0
    fi
    if ! grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err"; then
      cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2
      return 1
    fi
    sleep 300
    python3 -m harness.cli status >/dev/null
  done
}

wait_terminal() {
  local run_id="$1" state
  while true; do
    sleep 300
    python3 -m harness.cli status --run "$run_id" >/dev/null
    state="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$run_id/status.json")"
    case "$state" in
      SUCCEEDED) return 0 ;;
      FAILED|STOPPED|BLOCKED) echo "$run_id ended in $state" >&2; return 1 ;;
    esac
  done
}

TRAIN_RUNS=(
  "20260807-042119_cifar10-gmean-matched-head-20epoch"
  "20260807-042119_cifar10-raw-first-matched-head-20epoch"
)
PROBE_RUNS=(
  "20260807-052129_cifar10-gmean-matched-head-20epoch-linear-probe"
  "20260807-052130_cifar10-raw-first-matched-head-20epoch-linear-probe"
)
NAMES=("gmean-matched-head" "raw-first-matched-head")
OVERRIDES=(
  '{"model":{"parent_aggregation":"mean","f_head_hidden_dims":[2552]},"trainer":{"max_epochs":20},"probe":{"max_epochs":20}}'
  '{"model":{"parent_aggregation":"first","f_head_hidden_dims":[2552]},"trainer":{"max_epochs":20},"probe":{"max_epochs":20}}'
)
SEEDS=("20260832" "20260833")
RUNS=()

for index in 0 1; do
  train_checkpoint="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "runs/${TRAIN_RUNS[$index]}/artifacts/train_result.json")"
  probe_checkpoint="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["probe_checkpoint"])' "runs/${PROBE_RUNS[$index]}/artifacts/probe_result.json")"
  command="$PYTHON -m fmca_av.cli corruption-eval --config $CONFIG"
  command+=" --checkpoint $train_checkpoint --probe-checkpoint $probe_checkpoint --root $ROOT"
  command+=' --output "$FMCA_HARNESS_RUN_DIR/artifacts/cifar10c.json"'
  command+=" --device cuda --batch-size 256 --workers 0"
  run_id="$(submit_with_capacity_wait --name "cifar10-${NAMES[$index]}-20epoch-cifar10c" --gpus 1 -- env \
    "FMCA_CONFIG_OVERRIDES=${OVERRIDES[$index]}" "FMCA_SEED_OVERRIDE=${SEEDS[$index]}" \
    bash -lc "$command")"
  RUNS+=("$run_id")
  printf '%s\t%s\n' "${NAMES[$index]}" "$run_id" >> "$SUBMITTED"
done

for run_id in "${RUNS[@]}"; do wait_terminal "$run_id"; done
python3 -m harness.cli submit --name summarize-matched-cifar10-experiments-final --gpus 0 -- \
  "$PYTHON" scripts/summarize_results.py
