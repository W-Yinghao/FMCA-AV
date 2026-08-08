#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
TRAIN_RUN="20260807-052202_imagenet1k-lightning-32step-smoke"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"

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

# Do not poll sooner than the globally selected five-minute interval.
sleep 300
python3 -m harness.cli status --run "$TRAIN_RUN" >/dev/null
state="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$TRAIN_RUN/status.json")"
[[ "$state" == SUCCEEDED ]] || { echo "$TRAIN_RUN is $state" >&2; exit 1; }
checkpoint="$($PYTHON -c 'import json,sys; result=json.load(open(sys.argv[1])); print(result.get("best_checkpoint") or result.get("last_checkpoint") or "")' "runs/$TRAIN_RUN/artifacts/train_result.json")"
command="$PYTHON -m scripts.run_voc_multilabel_probe --config configs/ssl/imagenet1k_smoke.json"
command+=" --checkpoint $checkpoint --root /projects/EEG-foundation-model/yinghao/FMCA-AV/voc"
command+=' --output "$FMCA_HARNESS_RUN_DIR/artifacts"'
command+=" --epochs 3 --batch-size 128 --workers 8"
run_id="$(submit_with_capacity_wait --name imagenet1k-32step-voc2007-multilabel-smoke \
  --gpus 1 --profile imagenet -- bash -lc "$command")"
printf 'voc2007-multilabel\t%s\n' "$run_id" > "$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
