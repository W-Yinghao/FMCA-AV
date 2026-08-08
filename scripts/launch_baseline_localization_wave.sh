#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"; SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"; touch "$SUBMITTED"
existing_run_id() {
  awk -F '\t' -v method="$1" -v dataset="$2" \
    '$1 == method && $2 == dataset { value=$3 } END { if (value != "") print value }' "$SUBMITTED"
}
state_of() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$1/status.json"; }
wait_terminal() {
  local run_id="$1" state
  while true; do python3 -m harness.cli status --run "$run_id" >/dev/null; state="$(state_of "$run_id")"; case "$state" in SUCCEEDED|FAILED|STOPPED|BLOCKED) return 0;; *) sleep 300;; esac; done
}
harness_with_capacity_wait() {
  local output
  while true; do
    if output="$(python3 -m harness.cli "$@" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then printf '%s\n' "$output"; return 0; fi
    if ! grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err"; then cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2; return 1; fi
    sleep 300; python3 -m harness.cli status >/dev/null
  done
}

latest_named_run() {
  "$PYTHON" -c 'import json,sys
d=json.load(open("harness/state/jobs.json"))["jobs"]
rows=[v for v in d.values() if v.get("name")==sys.argv[1]]
print(max(rows,key=lambda v:v.get("created_at","")).get("run_id","") if rows else "")' "$1"
}

wait_children() {
  local pending run_id state
  while true; do
    sleep 300; python3 -m harness.cli status >/dev/null; pending=0
    while IFS=$'\t' read -r _method _dataset run_id; do
      [[ -n "$run_id" ]] || continue; state="$(state_of "$run_id")"
      case "$state" in SUCCEEDED) ;; FAILED|STOPPED|BLOCKED) echo "child $run_id ended in $state" >&2; return 1;; *) pending=1;; esac
    done < <(awk -F '\t' '{ latest[$1 FS $2]=$0 } END { for (key in latest) print latest[key] }' "$SUBMITTED")
    [[ "$pending" -eq 0 ]] && return 0
  done
}

sleep 300
CROSS_WATCHER="$(latest_named_run launch-cross-dataset-baseline-wave-fixed)"
[[ -n "$CROSS_WATCHER" ]] || { echo "cross-dataset baseline watcher missing" >&2; exit 1; }
CROSS_TSV="runs/$CROSS_WATCHER/artifacts/submitted_jobs.tsv"
wait_terminal "$CROSS_WATCHER"
[[ "$(state_of "$CROSS_WATCHER")" == SUCCEEDED ]] || { echo "$CROSS_WATCHER failed" >&2; exit 1; }
METHODS=(simclr vicreg dino)
for method in "${METHODS[@]}"; do
  source="$(awk -F '\t' -v method="$method" '$1=="imagenet100" && $2==method {value=$3} END {if(value!="") print value}' "$CROSS_TSV")"
  [[ -n "$source" ]] || { echo "missing ImageNet100 $method source" >&2; exit 1; }
  wait_terminal "$source"; [[ "$(state_of "$source")" == SUCCEEDED ]] || { echo "$source failed" >&2; exit 1; }
  checkpoint="$($PYTHON -c 'import json,sys; result=json.load(open(sys.argv[1])); print(result.get("best_checkpoint") or result.get("last_checkpoint") or "")' "runs/$source/artifacts/train_result.json")"
  override="{\"experiment\":{\"method\":\"$method\"}}"
  for dataset in cub voc imagenet; do
    prior="$(existing_run_id "$method" "$dataset")"
    if [[ -n "$prior" ]]; then
      prior_state="$(state_of "$prior")"
      case "$prior_state" in
        FAILED|STOPPED|BLOCKED)
          run_id="$(harness_with_capacity_wait retry --run "$prior")"
          printf '%s\t%s\t%s\n' "$method" "$dataset" "$run_id" >> "$SUBMITTED";;
      esac
      continue
    fi
    case "$dataset" in
      cub) root=/projects/EEG-foundation-model/yinghao/FMCA-AV/cub; extra="";;
      voc) root=/projects/EEG-foundation-model/yinghao/FMCA-AV/voc/VOC2012; extra="";;
      imagenet) root=/projects/EEG-foundation-model/yinghao/FMCA-AV/imagenet/ILSVRC; extra="--labels /projects/common/imagenet/LOC_val_solution.csv";;
    esac
    command="$PYTHON -m scripts.run_baseline_localization --config configs/ssl/imagenet100_smoke.json --checkpoint $checkpoint"
    command+=" --dataset $dataset --root $root --samples 100 $extra"
    command+=' --output "$FMCA_HARNESS_RUN_DIR/artifacts/localization.json"'
    run_id="$(harness_with_capacity_wait submit --name "imagenet100-${method}-${dataset}-localization" --gpus 1 --profile imagenet -- env "FMCA_CONFIG_OVERRIDES=$override" bash -lc "$command")"
    printf '%s\t%s\t%s\n' "$method" "$dataset" "$run_id" >> "$SUBMITTED"
  done
  if [[ "$method" == dino ]]; then
    prior="$(existing_run_id dino cub_randomized)"
    if [[ -n "$prior" ]]; then
      prior_state="$(state_of "$prior")"
      if [[ "$prior_state" == FAILED || "$prior_state" == STOPPED || "$prior_state" == BLOCKED ]]; then
        run_id="$(harness_with_capacity_wait retry --run "$prior")"
        printf 'dino\tcub_randomized\t%s\n' "$run_id" >> "$SUBMITTED"
      fi
      continue
    fi
    command="$PYTHON -m scripts.run_baseline_localization --config configs/ssl/imagenet100_smoke.json --checkpoint $checkpoint --dataset cub"
    command+=' --root /projects/EEG-foundation-model/yinghao/FMCA-AV/cub --samples 100 --randomize-backbone --output "$FMCA_HARNESS_RUN_DIR/artifacts/localization.json"'
    run_id="$(harness_with_capacity_wait submit --name imagenet100-dino-randomized-cub-localization --gpus 1 --profile imagenet -- env "FMCA_CONFIG_OVERRIDES=$override" bash -lc "$command")"
    printf 'dino\tcub_randomized\t%s\n' "$run_id" >> "$SUBMITTED"
  fi
done
wait_children
