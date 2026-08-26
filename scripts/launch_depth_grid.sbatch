#!/bin/bash
# (start, end) depth grid.
# Usage: sbatch launch_depth_grid.sbatch <model> <weights> <dataset> <out> [seed]
#SBATCH --job-name=depthgrid
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=runs/depthgrid_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
"$PYTHON" scripts/run_depth_grid.py \
  --model "$1" --weights "$2" --dataset "$3" \
  --root "${ROOT:-/projects/EEG-foundation-model/yinghao/FMCA-AV/$3}" \
  --samples 8000 --seed "${5:-1}" --out "$4"
