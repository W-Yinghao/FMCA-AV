#!/usr/bin/env bash
set -euo pipefail
[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
DOWNLOAD_RUN="20260807-065950_dataset-robustness-relocated"
TRAIN_RUN="20260807-052202_imagenet1k-lightning-32step-smoke"
PROBE_RUN="20260807-053349_imagenet1k-32step-smoke-linear-probe"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"; SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"; : > "$SUBMITTED"
state_of() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$1/status.json"; }
wait_success() {
  local run_id="$1" state
  while true; do
    python3 -m harness.cli status --run "$run_id" >/dev/null; state="$(state_of "$run_id")"
    case "$state" in SUCCEEDED) return 0;; FAILED|STOPPED|BLOCKED) echo "$run_id ended in $state" >&2; return 1;; *) sleep 300;; esac
  done
}
submit_with_capacity_wait() {
  local output
  while true; do
    if output="$(python3 -m harness.cli submit "$@" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then printf '%s\n' "$output"; return 0; fi
    grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" || { cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2; return 1; }
    sleep 300; python3 -m harness.cli status >/dev/null
  done
}
sleep 300; wait_success "$DOWNLOAD_RUN"
checkpoint="$($PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("best_checkpoint") or d.get("last_checkpoint") or "")' "runs/$TRAIN_RUN/artifacts/train_result.json")"
probe_checkpoint="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["probe_checkpoint"])' "runs/$PROBE_RUN/artifacts/probe_result.json")"
for suite in imagenet-c imagenet-r imagenet-a; do
  command="$PYTHON -m fmca_av.cli imagenet-robustness --config configs/ssl/imagenet1k_smoke.json"
  command+=" --checkpoint $checkpoint --probe-checkpoint $probe_checkpoint --root /projects/EEG-foundation-model/yinghao/FMCA-AV/robustness --suite $suite"
  command+=' --output "$FMCA_HARNESS_RUN_DIR/artifacts/imagenet_robustness.json" --device cuda --batch-size 256 --workers 8'
  run_id="$(submit_with_capacity_wait --name "imagenet1k-32step-${suite}-robustness-fixed" --gpus 1 --profile imagenet -- bash -lc "$command")"
  printf '%s\t%s\n' "$suite" "$run_id" >> "$SUBMITTED"
done
