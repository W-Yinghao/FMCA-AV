#!/usr/bin/env bash
set -euo pipefail
[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CHECKPOINT="runs/20260807-052202_imagenet1k-lightning-32step-smoke/artifacts/checkpoints/best-000-3.492188.ckpt"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"; SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"; : > "$SUBMITTED"
submit_with_capacity_wait() {
  local output
  while true; do
    if output="$(python3 -m harness.cli submit "$@" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then printf '%s\n' "$output"; return 0; fi
    if ! grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err"; then cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2; return 1; fi
    sleep 300; python3 -m harness.cli status >/dev/null
  done
}
sleep 300
run_id="$(submit_with_capacity_wait --name coco2017-fmca-detection-32step-smoke --gpus 1 --profile imagenet -- \
  "$PYTHON" -m scripts.run_coco_transfer --config configs/ssl/imagenet1k_smoke.json --checkpoint "$CHECKPOINT" \
  --root /projects/EEG-foundation-model/yinghao/FMCA-AV/coco --task detection --max-steps 32 --train-images 2000 --val-images 200)"
printf 'detection\t%s\n' "$run_id" >> "$SUBMITTED"
run_id="$(submit_with_capacity_wait --name coco2017-fmca-maskrcnn-16step-smoke --gpus 1 --profile imagenet -- \
  "$PYTHON" -m scripts.run_coco_transfer --config configs/ssl/imagenet1k_smoke.json --checkpoint "$CHECKPOINT" \
  --root /projects/EEG-foundation-model/yinghao/FMCA-AV/coco --task instance_segmentation --max-steps 16 --train-images 1000 --val-images 100)"
printf 'instance_segmentation\t%s\n' "$run_id" >> "$SUBMITTED"
