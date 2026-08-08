#!/usr/bin/env bash
set -euo pipefail
[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"; SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"; touch "$SUBMITTED"

state_of() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$1/status.json"; }
wait_success() {
  local run_id="$1" state
  while true; do
    python3 -m harness.cli status --run "$run_id" >/dev/null; state="$(state_of "$run_id")"
    case "$state" in SUCCEEDED) return 0;; FAILED|STOPPED|BLOCKED) echo "$run_id ended in $state" >&2; return 1;; *) sleep 300;; esac
  done
}
harness_with_capacity_wait() {
  local output
  while true; do
    if output="$(python3 -m harness.cli "$@" 2>"$FMCA_HARNESS_RUN_DIR/artifacts/submit.err")"; then printf '%s\n' "$output"; return 0; fi
    grep -q 'GPU limit exceeded' "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" || { cat "$FMCA_HARNESS_RUN_DIR/artifacts/submit.err" >&2; return 1; }
    sleep 300; python3 -m harness.cli status >/dev/null
  done
}

latest_named_run() {
  "$PYTHON" -c 'import json,sys
d=json.load(open("harness/state/jobs.json"))["jobs"]
rows=[v for v in d.values() if v.get("name")==sys.argv[1]]
print(max(rows,key=lambda v:v.get("created_at","")).get("run_id","") if rows else "")' "$1"
}

existing_run_id() { awk -F '\t' -v dataset="$1" '$1==dataset {value=$2} END {if(value!="") print value}' "$SUBMITTED"; }

wait_children() {
  local pending run_id state
  while true; do
    sleep 300; python3 -m harness.cli status >/dev/null; pending=0
    while IFS=$'\t' read -r _dataset run_id; do
      [[ -n "$run_id" ]] || continue; state="$(state_of "$run_id")"
      case "$state" in SUCCEEDED) ;; FAILED|STOPPED|BLOCKED) echo "child $run_id ended in $state" >&2; return 1;; *) pending=1;; esac
    done < <(awk -F '\t' '{latest[$1]=$0} END {for(key in latest) print latest[key]}' "$SUBMITTED")
    [[ "$pending" -eq 0 ]] && return 0
  done
}

sleep 300
TRAIN_WATCHER="$(latest_named_run launch-supervised-reference-wave)"
[[ -n "$TRAIN_WATCHER" ]] || { echo "supervised reference watcher missing" >&2; exit 1; }
TRAIN_TSV="runs/$TRAIN_WATCHER/artifacts/submitted_jobs.tsv"
wait_success "$TRAIN_WATCHER"
SOURCE="$(awk -F '\t' '$1=="imagenet1k" {value=$2} END {if(value!="") print value}' "$TRAIN_TSV")"
[[ -n "$SOURCE" ]] || { echo "ImageNet-1K supervised run missing" >&2; exit 1; }
wait_success "$SOURCE"
CHECKPOINT="$($PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("best_checkpoint") or d.get("last_checkpoint") or "")' "runs/$SOURCE/artifacts/supervised_result.json")"
[[ -n "$CHECKPOINT" ]] || { echo "ImageNet-1K supervised checkpoint missing" >&2; exit 1; }

for dataset in cub voc imagenet; do
  prior="$(existing_run_id "$dataset")"
  if [[ -n "$prior" ]]; then
    prior_state="$(state_of "$prior")"
    if [[ "$prior_state" == FAILED || "$prior_state" == STOPPED || "$prior_state" == BLOCKED ]]; then
      run_id="$(harness_with_capacity_wait retry --run "$prior")"
      printf '%s\t%s\n' "$dataset" "$run_id" >> "$SUBMITTED"
    fi
    continue
  fi
  args=()
  case "$dataset" in
    cub) root=/projects/EEG-foundation-model/yinghao/FMCA-AV/cub;;
    voc) root=/projects/EEG-foundation-model/yinghao/FMCA-AV/voc/VOC2012;;
    imagenet) root=/projects/EEG-foundation-model/yinghao/FMCA-AV/imagenet/ILSVRC; args=(--labels /projects/common/imagenet/LOC_val_solution.csv);;
  esac
  run_id="$(harness_with_capacity_wait submit --name "supervised-imagenet1k-${dataset}-localization" --gpus 1 --profile imagenet -- \
    "$PYTHON" -m scripts.run_supervised_localization --config configs/ssl/imagenet1k_smoke.json --checkpoint "$CHECKPOINT" \
    --dataset "$dataset" --root "$root" --samples 100 "${args[@]}")"
  printf '%s\t%s\n' "$dataset" "$run_id" >> "$SUBMITTED"
done
prior="$(existing_run_id cub_randomized)"
if [[ -n "$prior" ]]; then
  prior_state="$(state_of "$prior")"
  if [[ "$prior_state" == FAILED || "$prior_state" == STOPPED || "$prior_state" == BLOCKED ]]; then
    run_id="$(harness_with_capacity_wait retry --run "$prior")"
    printf 'cub_randomized\t%s\n' "$run_id" >> "$SUBMITTED"
  fi
else
  run_id="$(harness_with_capacity_wait submit --name supervised-imagenet1k-randomized-cub-localization --gpus 1 --profile imagenet -- \
    "$PYTHON" -m scripts.run_supervised_localization --config configs/ssl/imagenet1k_smoke.json --checkpoint "$CHECKPOINT" \
    --dataset cub --root /projects/EEG-foundation-model/yinghao/FMCA-AV/cub --samples 100 --randomize-backbone)"
  printf 'cub_randomized\t%s\n' "$run_id" >> "$SUBMITTED"
fi
wait_children
