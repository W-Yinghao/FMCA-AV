#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || {
  echo "This sweep submitter must run through the Slurm harness" >&2
  exit 3
}

PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG="configs/toy/finite_learning_reference.json"
START=1
END=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
: > "$SUBMITTED"

for replicate in $(seq "$START" "$END"); do
  seed=$((20260900 + replicate))
  while true; do
    if run_id="$(python3 -m harness.cli submit \
      --name "e1-finite-reference-seed-${seed}" \
      --gpus 0 -- env \
      "FMCA_SEED_OVERRIDE=${seed}" \
      bash scripts/run_fmca_pipeline.sh --config "$CONFIG")"; then
      break
    fi
    echo "Submission for seed $seed is temporarily unavailable; retrying in 300 seconds" >&2
    sleep 300
  done
  printf '%s\t%s\n' "$seed" "$run_id" >> "$SUBMITTED"
done
