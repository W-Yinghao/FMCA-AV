#!/usr/bin/env bash
set -euo pipefail
[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
sleep 300
while true; do
  if run_id="$(python3 -m harness.cli submit --name e10-gpu-complexity-full --gpus 1 --profile imagenet -- \
    /projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python -m scripts.run_complexity_benchmark \
    2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then
    printf '%s\n' "$run_id" > "$FMCA_HARNESS_RUN_DIR/artifacts/submitted_run.txt"; break
  fi
  grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" || { cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2; exit 1; }
  sleep 300; python3 -m harness.cli status >/dev/null
done
