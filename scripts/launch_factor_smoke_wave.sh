#!/usr/bin/env bash
set -euo pipefail

[[ -n "${FMCA_HARNESS_RUN_DIR:-}" ]] || { echo "run through Slurm harness" >&2; exit 3; }
PYTHON="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
PREPARE_RUN="20260807-053702_prepare-factor-memmaps"
mkdir -p "$FMCA_HARNESS_RUN_DIR/artifacts"
SUBMITTED="$FMCA_HARNESS_RUN_DIR/artifacts/submitted_jobs.tsv"
: > "$SUBMITTED"

refresh_state() {
  local run_id="$1"
  python3 -m harness.cli status --run "$run_id" >/dev/null
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "runs/$run_id/status.json"
}

wait_existing() {
  local run_id="$1" state
  while true; do
    state="$(refresh_state "$run_id")"
    case "$state" in
      SUCCEEDED) return 0 ;;
      FAILED|STOPPED|BLOCKED) echo "$run_id ended in $state" >&2; return 1 ;;
      *) sleep 300 ;;
    esac
  done
}

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

wait_new() {
  sleep 300
  wait_existing "$1"
}

sleep 300
wait_existing "$PREPARE_RUN"

VALIDATION="$(python3 -m harness.cli submit --name validate-factor-lightning-data --gpus 0 -- bash -lc \
  "$PYTHON - <<'PY'
from fmca_av.config import load_config
from fmca_av.data.factors import FactorDataModule
for path in ('configs/ssl/dsprites_smoke.json','configs/ssl/shapes3d_smoke.json','configs/ssl/smallnorb_smoke.json','configs/ssl/mpi3d_toy_smoke.json'):
    config=load_config(path)
    module=FactorDataModule(config['data'],config['seed'])
    module.setup()
    batch=next(iter(module.val_dataloader()))
    assert batch[0].ndim == 5 and batch[1].ndim == 2
    print(path,tuple(batch[0].shape),tuple(batch[1].shape))
PY")"
printf 'validation\t%s\n' "$VALIDATION" >> "$SUBMITTED"
wait_new "$VALIDATION"

CONFIGS=(dsprites shapes3d smallnorb mpi3d_toy)
TRAIN_RUNS=()
for name in "${CONFIGS[@]}"; do
  run_id="$(submit_with_capacity_wait --name "${name}-lightning-32step-smoke" --gpus 1 -- \
    bash scripts/run_fmca_pipeline.sh --config "configs/ssl/${name}_smoke.json")"
  TRAIN_RUNS+=("$run_id")
  printf 'train\t%s\t%s\n' "$name" "$run_id" >> "$SUBMITTED"
done

for index in "${!TRAIN_RUNS[@]}"; do
  run_id="${TRAIN_RUNS[$index]}"
  wait_new "$run_id"
  checkpoint="$($PYTHON -c 'import json,sys; result=json.load(open(sys.argv[1])); print(result.get("best_checkpoint") or result.get("last_checkpoint") or "")' "runs/$run_id/artifacts/train_result.json")"
  calibration="runs/$run_id/artifacts/calibration.pt"
  name="${CONFIGS[$index]}"
  command="$PYTHON -m scripts.run_factor_spectral_probe --config configs/ssl/${name}_smoke.json"
  command+=" --checkpoint $checkpoint --calibration $calibration"
  command+=' --output "$FMCA_HARNESS_RUN_DIR/artifacts/factor_probe.json"'
  command+=" --train-samples 5000 --test-samples 2000 --random-repeats 5 --rotation-repeats 2 --device cuda"
  probe_run="$(submit_with_capacity_wait --name "${name}-spectral-factor-probe-smoke" --gpus 1 -- bash -lc "$command")"
  printf 'probe\t%s\t%s\n' "$name" "$probe_run" >> "$SUBMITTED"
done
