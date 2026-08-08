#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
C10_WATCHER="20260807-060401_continue-cifar10-baseline-screening-fixed"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
touch "$SUBMITTED"

existing_run_id() {
  awk -F '\t' -v dataset="$1" -v method="$2" \
    '$1 == dataset && $2 == method { value=$3 } END { if (value != "") print value }' "$SUBMITTED"
}

harness_with_capacity_wait() {
  local output
  while true; do
    if output="$(python3 -m harness.cli "$@" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then printf '%s\n' "$output"; return 0; fi
    if ! grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err"; then cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2; return 1; fi
    sleep 300; python3 -m harness.cli status >/dev/null
  done
}

run_state() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$1/status.json"
}

repair_terminal_failure() {
  local dataset="$1" method="$2" prior="$3" state="$4" replacement
  case "$state" in
    FAILED|STOPPED|BLOCKED)
      replacement="$(harness_with_capacity_wait retry --run "$prior")"
      printf '%s\t%s\t%s\n' "$dataset" "$method" "$replacement" >> "$SUBMITTED"
      printf '%s\n' "$replacement"
      ;;
    *) printf '%s\n' "$prior";;
  esac
}

wait_children() {
  local failed=0 run_id state
  while true; do
    sleep 300
    python3 -m harness.cli status >/dev/null
    failed=0
    while IFS=$'\t' read -r _dataset _method run_id; do
      [[ -n "$run_id" ]] || continue
      state="$(run_state "$run_id")"
      case "$state" in
        SUCCEEDED) ;;
        FAILED|STOPPED|BLOCKED) echo "child $run_id ended in $state" >&2; return 1;;
        *) failed=1;;
      esac
    done < <(awk -F '\t' '{ latest[$1 FS $2]=$0 } END { for (key in latest) print latest[key] }' "$SUBMITTED")
    [[ "$failed" -eq 0 ]] && return 0
  done
}

wait_c10() {
  local state
  while true; do
    python3 -m harness.cli status --run "$C10_WATCHER" >/dev/null
    state="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$C10_WATCHER/status.json")"
    case "$state" in SUCCEEDED) return 0;; FAILED|STOPPED|BLOCKED) return 1;; *) sleep 300;; esac
  done
}

sleep 300
wait_c10
METHODS=(simclr barlow_twins vicreg spectral_contrastive fastsiam byol moco_v2 dino)
DATASETS=(cifar100 stl10 tinyimagenet200 imagenet100)
CONFIGS=(configs/ssl/cifar100_smoke.json configs/ssl/stl10_smoke.json configs/ssl/tinyimagenet200_smoke.json configs/ssl/imagenet100_smoke.json)
seed=20261300
for data_index in 0 1 2 3; do
  dataset="${DATASETS[$data_index]}"; config="${CONFIGS[$data_index]}"
  profile=(); [[ "$dataset" == imagenet100 ]] && profile=(--profile imagenet)
  for method in "${METHODS[@]}"; do
    seed=$((seed + 1))
    prior="$(existing_run_id "$dataset" "$method")"
    if [[ -n "$prior" ]]; then
      state="$(run_state "$prior")"
      replacement="$(repair_terminal_failure "$dataset" "$method" "$prior" "$state")"
      [[ "$replacement" != "$prior" ]] || continue
      continue
    fi
    override="{\"experiment\":{\"method\":\"$method\"}}"
    run_id="$(harness_with_capacity_wait submit --name "${dataset}-${method}-screening" --gpus 1 "${profile[@]}" -- env \
      "FMCA_CONFIG_OVERRIDES=$override" "FMCA_SEED_OVERRIDE=$seed" \
      "$PYTHON" -m fmca_av.baseline_cli train --config "$config")"
    printf '%s\t%s\t%s\n' "$dataset" "$method" "$run_id" >> "$SUBMITTED"
  done
done
wait_children
