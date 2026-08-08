#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG="configs/ssl/cifar10_baseline_smoke.json"
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

wait_all_success() {
  local state run_id active
  while true; do
    python3 -m harness.cli status >/dev/null
    active=0
    for run_id in "$@"; do
      state="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$run_id/status.json")"
      case "$state" in
        SUCCEEDED) ;;
        FAILED|STOPPED|BLOCKED) echo "$run_id ended in $state" >&2; return 1 ;;
        *) active=1 ;;
      esac
    done
    [[ "$active" -eq 0 ]] && return 0
    sleep 300
  done
}

METHODS=(simclr barlow_twins vicreg spectral_contrastive fastsiam byol moco_v2 dino)
SEEDS=(20260850 20260851 20260852 20260853 20260854 20260855 20260856 20260857)
TRAIN_RUNS=("20260807-053905_cifar10-simclr-5epoch-screening")

# The initial sleep guarantees that this watcher never introduces a sub-300s status refresh.
sleep 300
for index in 1 2 3 4 5 6 7; do
  method="${METHODS[$index]}"
  override="{\"experiment\":{\"method\":\"$method\"}}"
  run_id="$(submit_with_capacity_wait --name "cifar10-${method}-5epoch-screening" --gpus 1 -- env \
    "FMCA_CONFIG_OVERRIDES=$override" "FMCA_SEED_OVERRIDE=${SEEDS[$index]}" \
    "$PYTHON" -m fmca_av.baseline_cli train --config "$CONFIG")"
  TRAIN_RUNS+=("$run_id")
  printf 'train\t%s\t%s\n' "$method" "$run_id" >> "$SUBMITTED"
done

sleep 300
PROBE_RUNS=()
wait_all_success "${TRAIN_RUNS[@]}"
for index in "${!METHODS[@]}"; do
  run_id="${TRAIN_RUNS[$index]}"
  checkpoint="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "runs/$run_id/artifacts/train_result.json")"
  method="${METHODS[$index]}"
  override="{\"experiment\":{\"method\":\"$method\"}}"
  probe_run="$(submit_with_capacity_wait --name "cifar10-${method}-5epoch-linear-probe" --gpus 1 -- env \
    "FMCA_CONFIG_OVERRIDES=$override" "FMCA_SEED_OVERRIDE=${SEEDS[$index]}" \
    "$PYTHON" -m fmca_av.baseline_cli linear-probe --config "$CONFIG" --checkpoint "$checkpoint")"
  PROBE_RUNS+=("$probe_run")
  printf 'probe\t%s\t%s\n' "$method" "$probe_run" >> "$SUBMITTED"
done

sleep 300
wait_all_success "${PROBE_RUNS[@]}"
python3 -m harness.cli submit --name summarize-cifar10-baseline-screening --gpus 0 -- \
  "$PYTHON" scripts/summarize_results.py
