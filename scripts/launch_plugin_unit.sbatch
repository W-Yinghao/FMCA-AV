#!/bin/bash
# Plug-in study unit. Usage: sbatch launch_plugin_unit.sbatch <row> <seed> [probe]
#   row in {barlow_twins,vicreg,frossl}_{base,plugin}
#SBATCH --job-name=plugin
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=runs/plugin_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
ARGS=(--config "configs/plugin/$1.json" --seed "$2")
if [ "${3:-full}" = "probe" ]; then ARGS+=(--probe-mode); fi
"$PYTHON" scripts/run_plugin_unit.py "${ARGS[@]}"
