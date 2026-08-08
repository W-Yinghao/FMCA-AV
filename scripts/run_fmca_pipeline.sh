#!/usr/bin/env bash
set -euo pipefail

CONFIG=""
FMCA_PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --python) FMCA_PYTHON="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$CONFIG" ]] || { echo "--config is required" >&2; exit 2; }
[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || {
  echo "This pipeline must be launched through the Slurm harness" >&2
  exit 3
}

"$FMCA_PYTHON" -m fmca_av.cli train --config "$CONFIG"
ARTIFACTS="$FMCA_HARNESS_RUN_DIR/artifacts"
CHECKPOINT="$($FMCA_PYTHON -c 'import json,sys; result=json.load(open(sys.argv[1])); print(result.get("best_checkpoint") or result.get("last_checkpoint") or "")' "$ARTIFACTS/train_result.json")"
[[ -n "$CHECKPOINT" ]] || {
  echo "Training produced neither a best nor a last checkpoint" >&2
  exit 1
}
DEVICE="$($FMCA_PYTHON -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"

"$FMCA_PYTHON" -m fmca_av.cli calibrate \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$ARTIFACTS/calibration.pt" \
  --device "$DEVICE"

"$FMCA_PYTHON" -m fmca_av.cli evaluate \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --calibration "$ARTIFACTS/calibration.pt" \
  --output "$ARTIFACTS/evaluation.json" \
  --device "$DEVICE"
