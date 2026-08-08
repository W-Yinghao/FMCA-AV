#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
SOURCE_WATCHER="20260807-054148_launch-factor-smoke-wave"
SOURCE_TSV="runs/$SOURCE_WATCHER/artifacts/submitted_jobs.tsv"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
: > "$SUBMITTED"

state_of() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$1/status.json"
}

refresh_wait_terminal() {
  local run_id="$1" state
  while true; do
    python3 -m harness.cli status --run "$run_id" >/dev/null
    state="$(state_of "$run_id")"
    case "$state" in SUCCEEDED|FAILED|STOPPED|BLOCKED) return 0;; *) sleep 300;; esac
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

retry_with_capacity_wait() {
  local source="$1" output
  while true; do
    if output="$(python3 -m harness.cli retry --run "$source" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/retry.err")"; then printf '%s\n' "$output"; return 0; fi
    if ! grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/retry.err"; then cat "$FMCA_HARNESS_RUN_DIR/artifacts/retry.err" >&2; return 1; fi
    sleep 300; python3 -m harness.cli status >/dev/null
  done
}

sleep 300
refresh_wait_terminal "$SOURCE_WATCHER"
DATASETS=(dsprites shapes3d smallnorb mpi3d_toy)
TRAIN_RUNS=()
for dataset in "${DATASETS[@]}"; do
  source="$(awk -F '\t' -v name="$dataset" '$1=="train" && $2==name {value=$3} END {print value}' "$SOURCE_TSV" 2>/dev/null || true)"
  if [[ -z "$source" ]]; then
    run_id="$(submit_with_capacity_wait --name "${dataset}-lightning-32step-smoke" --gpus 1 -- bash scripts/run_fmca_pipeline.sh --config "configs/ssl/${dataset}_smoke.json")"
    action="submitted"
  else
    state="$(state_of "$source")"
    if [[ "$state" == SUCCEEDED ]]; then run_id="$source"; action="reused"
    elif [[ "$state" == FAILED || "$state" == STOPPED || "$state" == BLOCKED ]]; then run_id="$(retry_with_capacity_wait "$source")"; action="retried"
    else run_id="$source"; action="reused-active"; fi
  fi
  TRAIN_RUNS+=("$run_id")
  printf 'train\t%s\t%s\t%s\n' "$dataset" "$run_id" "$action" >> "$SUBMITTED"
done

sleep 300
for run_id in "${TRAIN_RUNS[@]}"; do
  refresh_wait_terminal "$run_id"
  [[ "$(state_of "$run_id")" == SUCCEEDED ]] || { echo "$run_id did not succeed" >&2; exit 1; }
done

for index in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$index]}"; train_run="${TRAIN_RUNS[$index]}"
  checkpoint="$($PYTHON -c 'import json,sys; result=json.load(open(sys.argv[1])); print(result.get("best_checkpoint") or result.get("last_checkpoint") or "")' "runs/$train_run/artifacts/train_result.json")"
  calibration="runs/$train_run/artifacts/calibration.pt"
  command="$PYTHON -m scripts.run_factor_spectral_probe --config configs/ssl/${dataset}_smoke.json"
  command+=" --checkpoint $checkpoint --calibration $calibration"
  command+=' --output "$FMCA_HARNESS_RUN_DIR/artifacts/factor_probe.json"'
  command+=" --train-samples 5000 --test-samples 2000 --random-repeats 5 --rotation-repeats 2 --device cuda"
  probe_run="$(submit_with_capacity_wait --name "${dataset}-spectral-factor-probe-smoke" --gpus 1 -- bash -lc "$command")"
  printf 'probe\t%s\t%s\n' "$dataset" "$probe_run" >> "$SUBMITTED"
done
