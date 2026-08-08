#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
VALIDATION="20260807-055339_validate-convnext-vit-small-architectures"
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
    sleep 300; python3 -m harness.cli status >/dev/null
  done
}

sleep 300
python3 -m harness.cli status --run "$VALIDATION" >/dev/null
state="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$VALIDATION/status.json")"
[[ "$state" == SUCCEEDED ]] || { echo "$VALIDATION is $state" >&2; exit 1; }

DATASETS=(cifar100 cifar100 imagenet100 imagenet100)
BACKBONES=(convnext_tiny vit_s_16 convnext_tiny vit_s_16)
CONFIGS=(configs/ssl/cifar100_smoke.json configs/ssl/cifar100_smoke.json configs/ssl/imagenet100_smoke.json configs/ssl/imagenet100_smoke.json)
for index in 0 1 2 3; do
  dataset="${DATASETS[$index]}"; backbone="${BACKBONES[$index]}"; config="${CONFIGS[$index]}"
  batch=16; profile=()
  if [[ "$dataset" == imagenet100 ]]; then batch=32; profile=(--profile imagenet); fi
  override="{\"model\":{\"backbone\":\"$backbone\"},\"data\":{\"batch_size\":$batch,\"augmentation\":{\"size\":224}},\"trainer\":{\"max_epochs\":1,\"max_steps\":32,\"limit_val_batches\":8}}"
  run_id="$(submit_with_capacity_wait --name "${dataset}-${backbone}-32step-smoke" --gpus 1 "${profile[@]}" -- env \
    "FMCA_CONFIG_OVERRIDES=$override" bash scripts/run_fmca_pipeline.sh --config "$config")"
  printf '%s\t%s\t%s\n' "$dataset" "$backbone" "$run_id" >> "$SUBMITTED"
done
