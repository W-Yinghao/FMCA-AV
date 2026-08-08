#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || {
  echo "This continuation must run through the Slurm harness" >&2
  exit 3
}

PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG="configs/ssl/imagenet1k_smoke.json"
TRAIN_RUN="20260807-052202_imagenet1k-lightning-32step-smoke"
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
CHECKPOINT="$(read_json_field "runs/$TRAIN_RUN/artifacts/train_result.json" best_checkpoint)"

PROBE_RUN="$(python3 -m harness.cli submit \
  --name imagenet1k-32step-smoke-linear-probe --gpus 1 --profile imagenet -- \
  "$PYTHON" -m fmca_av.cli linear-probe --config "$CONFIG" --checkpoint "$CHECKPOINT")"
printf 'linear-probe\t%s\n' "$PROBE_RUN" >> "$SUBMITTED"
wait_success "$PROBE_RUN"

command="$PYTHON -m fmca_av.cli knn --config $CONFIG --checkpoint $CHECKPOINT"
command+=' --output "$FMCA_HARNESS_RUN_DIR/artifacts/knn.json"'
command+=" --device cuda --batch-size 128 --workers 8 --bank-limit 10000 --bank-chunk-size 4096"
KNN_RUN="$(python3 -m harness.cli submit \
  --name imagenet1k-32step-smoke-knn10k --gpus 1 --profile imagenet -- bash -lc "$command")"
printf 'knn-10k\t%s\n' "$KNN_RUN" >> "$SUBMITTED"
wait_success "$KNN_RUN"

python3 -m harness.cli submit --name summarize-imagenet1k-smoke --gpus 0 \
  -- "$PYTHON" scripts/summarize_results.py
