#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG="configs/ssl/imagenet1k_smoke.json"
TRAIN_RUN="20260807-052202_imagenet1k-lightning-32step-smoke"
VALIDATION_RUN="20260807-054937_validate-factor-transfer-localization-syntax"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
: > "$SUBMITTED"

submit_with_capacity_wait() {
  local output
  while true; do
    if output="$(python3 -m harness.cli submit "$@" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then
      printf '%s\n' "$output"; return 0
    fi
    if ! grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err"; then
      cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2; return 1
    fi
    sleep 300
    python3 -m harness.cli status >/dev/null
  done
}

sleep 300
python3 -m harness.cli status >/dev/null
state="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$VALIDATION_RUN/status.json")"
[[ "$state" == SUCCEEDED ]] || { echo "$VALIDATION_RUN is $state" >&2; exit 1; }
checkpoint="$($PYTHON -c 'import json,sys; result=json.load(open(sys.argv[1])); print(result.get("best_checkpoint") or result.get("last_checkpoint") or "")' "runs/$TRAIN_RUN/artifacts/train_result.json")"
calibration="runs/$TRAIN_RUN/artifacts/calibration.pt"

submit_map() {
  local name="$1" dataset="$2" root="$3" extra="$4"
  local command run_id
  command="$PYTHON -m scripts.run_dependence_localization --config $CONFIG --checkpoint $checkpoint"
  command+=" --calibration $calibration --dataset $dataset --root $root --samples 100 --modes 8 $extra"
  command+=' --output "$FMCA_HARNESS_RUN_DIR/artifacts/localization.json"'
  run_id="$(submit_with_capacity_wait --name "$name" --gpus 1 --profile imagenet -- bash -lc "$command")"
  printf '%s\t%s\n' "$name" "$run_id" >> "$SUBMITTED"
}

submit_map imagenet1k-32step-cub-dependence-localization cub \
  /projects/EEG-foundation-model/yinghao/FMCA-AV/cub ""
submit_map imagenet1k-32step-voc-dependence-localization voc \
  /projects/EEG-foundation-model/yinghao/FMCA-AV/voc/VOC2012 ""
submit_map imagenet1k-32step-imagenet-dependence-localization imagenet \
  /projects/EEG-foundation-model/yinghao/FMCA-AV/imagenet/ILSVRC \
  "--labels /projects/common/imagenet/LOC_val_solution.csv"
submit_map imagenet1k-randomized-cub-localization-sanity cub \
  /projects/EEG-foundation-model/yinghao/FMCA-AV/cub "--randomize-backbone"
