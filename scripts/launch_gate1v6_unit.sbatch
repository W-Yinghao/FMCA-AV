#!/bin/bash
# Gate 1 v6 unit (differentiable whitening for hierarchy rows).
# Usage: sbatch launch_gate1v6_unit.sbatch <variant> <seed> [probe]
#SBATCH --job-name=gate1v6
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=runs/gate1v6_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
ARGS=(--variant "$1" --seed "$2" --config-dir configs/gate_v6 --output-root results/gate1/gate1_20260818_v6)
if [ "${3:-full}" = "probe" ]; then ARGS+=(--probe-mode); fi
"$PYTHON" scripts/run_gate1_unit.py "${ARGS[@]}"
