#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
: > "$SUBMITTED"

submit_with_capacity_wait() {
  local output
  while true; do
    if output="$(python3 -m harness.cli submit "$@" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then printf '%s\n' "$output"; return 0; fi
    if ! grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err"; then cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2; return 1; fi
    sleep 300; python3 -m harness.cli status >/dev/null
  done
}

read_checkpoint() {
  "$PYTHON" -c 'import json,sys; result=json.load(open(sys.argv[1])); print(result.get("best_checkpoint") or result.get("last_checkpoint") or "")' "runs/$1/artifacts/train_result.json"
}

sleep 300
C10_RUN="20260807-042119_cifar10-gmean-matched-head-20epoch"
C100_RUN="20260807-043435_cifar100-lightning-20epoch"
IN100_RUN="20260807-052850_imagenet100-lightning-32step-smoke"
C10_CHECKPOINT="$(read_checkpoint "$C10_RUN")"
C100_CHECKPOINT="$(read_checkpoint "$C100_RUN")"
IN100_CHECKPOINT="$(read_checkpoint "$IN100_RUN")"

for fraction in 0.01 0.1 1.0; do
  tag="${fraction//./p}"
  override="{\"model\":{\"parent_aggregation\":\"mean\",\"f_head_hidden_dims\":[2552]},\"trainer\":{\"max_epochs\":20},\"probe\":{\"max_epochs\":20,\"label_fraction\":$fraction}}"
  run_id="$(submit_with_capacity_wait --name "cifar10-gmean-lowlabel-${tag}" --gpus 1 -- env \
    "FMCA_CONFIG_OVERRIDES=$override" "$PYTHON" -m fmca_av.cli linear-probe \
    --config configs/ssl/cifar10_paper_concat_smoke.json --checkpoint "$C10_CHECKPOINT")"
  printf 'cifar10\t%s\t%s\n' "$fraction" "$run_id" >> "$SUBMITTED"

  override="{\"trainer\":{\"max_epochs\":20},\"probe\":{\"max_epochs\":20,\"label_fraction\":$fraction}}"
  run_id="$(submit_with_capacity_wait --name "cifar100-lowlabel-${tag}" --gpus 1 -- env \
    "FMCA_CONFIG_OVERRIDES=$override" "$PYTHON" -m fmca_av.cli linear-probe \
    --config configs/ssl/cifar100_smoke.json --checkpoint "$C100_CHECKPOINT")"
  printf 'cifar100\t%s\t%s\n' "$fraction" "$run_id" >> "$SUBMITTED"

  override="{\"probe\":{\"max_epochs\":5,\"label_fraction\":$fraction,\"limit_train_batches\":32,\"limit_val_batches\":8,\"limit_test_batches\":8}}"
  run_id="$(submit_with_capacity_wait --name "imagenet100-lowlabel-${tag}-smoke" --gpus 1 --profile imagenet -- env \
    "FMCA_CONFIG_OVERRIDES=$override" "$PYTHON" -m fmca_av.cli linear-probe \
    --config configs/ssl/imagenet100_smoke.json --checkpoint "$IN100_CHECKPOINT")"
  printf 'imagenet100\t%s\t%s\n' "$fraction" "$run_id" >> "$SUBMITTED"
done
