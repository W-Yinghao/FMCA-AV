#!/bin/bash
# Gate 1 v4 unit (photometric add-one hierarchy).
# Usage: sbatch launch_gate1v4_unit.sbatch <variant> <seed> [probe]
#SBATCH --job-name=gate1v4
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=runs/gate1v4_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
VARIANT="$1"
SEED="$2"
MODE="${3:-full}"
ARGS=(--variant "$VARIANT" --seed "$SEED" --config-dir configs/gate_v4 --output-root results/gate1/gate1_20260817_v4)
if [ "$MODE" = "probe" ]; then ARGS+=(--probe-mode); fi
"$PYTHON" scripts/run_gate1_unit.py "${ARGS[@]}"
