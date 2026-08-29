#!/bin/bash
# Generic gate unit launcher.
# Usage: sbatch launch_gate1_generic.sbatch <config-dir> <output-root> <variant> <seed> [probe]
#SBATCH --job-name=gate1x
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=runs/gate1x_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
ARGS=(--variant "$3" --seed "$4" --config-dir "$1" --output-root "$2")
if [ "${5:-full}" = "probe" ]; then ARGS+=(--probe-mode); fi
"$PYTHON" scripts/run_gate1_unit.py "${ARGS[@]}"
