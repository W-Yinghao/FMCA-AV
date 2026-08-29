#!/bin/bash
# Gate 1 unit: one (variant, seed) on one GPU.
# Prereg: prereg/GATE1_CIFAR10_STRUCTURE_PREREG_FROZEN_20260816.md
# Usage: sbatch launch_gate1_unit.sbatch <variant> <seed> [probe]
# node51 (L40S) is excluded: uncorrectable ECC failures on record.
#SBATCH --job-name=gate1
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=runs/gate1_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
VARIANT="$1"
SEED="$2"
MODE="${3:-full}"

if [ "$MODE" = "probe" ]; then
  "$PYTHON" scripts/run_gate1_unit.py --variant "$VARIANT" --seed "$SEED" --probe-mode
else
  "$PYTHON" scripts/run_gate1_unit.py --variant "$VARIANT" --seed "$SEED"
fi
