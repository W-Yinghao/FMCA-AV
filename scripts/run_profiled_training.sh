#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--" ]]; then
  shift
fi
if [[ $# -eq 0 ]]; then
  echo "usage: $0 -- <training command> [args...]" >&2
  exit 2
fi
: "${FMCA_HARNESS_RUN_DIR:?run through the harness}"

artifacts="$FMCA_HARNESS_RUN_DIR/artifacts"
mkdir -p "$artifacts"

gpu_selector=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  gpu_selector=(-i "$CUDA_VISIBLE_DEVICES")
fi
printf '%s\n' "${CUDA_VISIBLE_DEVICES:-}" > "$artifacts/allocated_gpu_ids.txt"

nvidia-smi -L > "$artifacts/gpu_inventory.txt"
nvidia-smi topo -m > "$artifacts/gpu_topology.txt" 2>&1 || true
nvidia-smi "${gpu_selector[@]}" -q -d MEMORY,PERFORMANCE,POWER,CLOCK,COMPUTE \
  > "$artifacts/gpu_initial.txt" 2>&1 || true
lscpu > "$artifacts/cpu_inventory.txt" 2>&1 || true
df -T "$PWD" "$(dirname "$FMCA_HARNESS_RUN_DIR")" \
  > "$artifacts/filesystem_inventory.txt" 2>&1 || true

(
  echo "sample_time,index,name,pstate,gpu_util_pct,memory_util_pct,memory_used_mib,memory_total_mib,power_w,sm_clock_mhz,memory_clock_mhz,temperature_c"
  while true; do
    sample_time="$(date --iso-8601=seconds)"
    nvidia-smi "${gpu_selector[@]}" \
      --query-gpu=index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,temperature.gpu \
      --format=csv,noheader,nounits | sed "s/^/$sample_time,/"
    sleep 2
  done
) > "$artifacts/gpu_samples.csv" 2> "$artifacts/gpu_sampler.stderr" &
gpu_sampler_pid=$!

vmstat 2 > "$artifacts/vmstat.log" 2>&1 &
vmstat_pid=$!

cleanup() {
  kill "$gpu_sampler_pid" "$vmstat_pid" 2>/dev/null || true
  wait "$gpu_sampler_pid" "$vmstat_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

set +e
/usr/bin/time -v -o "$artifacts/resource_usage.txt" "$@"
exit_code=$?
set -e
cleanup
trap - EXIT INT TERM
exit "$exit_code"
