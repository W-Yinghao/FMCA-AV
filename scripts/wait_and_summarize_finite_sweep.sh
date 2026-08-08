#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || {
  echo "This summary must run through the Slurm harness" >&2
  exit 3
}

PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
SUBMISSION_FILES=(
  "runs/20260807-043316_submit-e1-finite-20seed-sweep/artifacts/submitted_jobs.tsv"
  "runs/20260807-043605_submit-e1-finite-seeds-15-20/artifacts/submitted_jobs.tsv"
)

for submission_file in "${SUBMISSION_FILES[@]}"; do
  while [[ ! -f "$submission_file" ]]; do sleep 300; done
  while IFS=$'\t' read -r seed run_id; do
    [[ -n "$run_id" ]] || continue
    while true; do
      python3 -m harness.cli status --run "$run_id" >/dev/null
      state="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' \
        "runs/$run_id/status.json")"
      case "$state" in
        SUCCEEDED) break ;;
        FAILED|STOPPED|BLOCKED)
          echo "seed $seed run $run_id ended in $state; leaving it missing" >&2
          break
          ;;
        *) sleep 300 ;;
      esac
    done
  done < "$submission_file"
done

mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
"$PYTHON" scripts/summarize_finite_sweep.py \
  --output "$FMCA_HARNESS_RUN_DIR/artifacts/finite_seed_summary.json"
