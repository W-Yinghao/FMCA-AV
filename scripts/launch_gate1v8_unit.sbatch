#!/bin/bash
# FINAL gate v8 unit. Usage: sbatch launch_gate1v8_unit.sbatch <variant> <seed> [probe]
#SBATCH --job-name=gate1v8
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=runs/gate1v8_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
ARGS=(--variant "$1" --seed "$2" --config-dir configs/gate_v8 --output-root results/gate1/gate1_20260820_v8)
if [ "${3:-full}" = "probe" ]; then ARGS+=(--probe-mode); fi
"$PYTHON" scripts/run_gate1_unit.py "${ARGS[@]}"
