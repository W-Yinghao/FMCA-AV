#!/bin/bash
# Pilot A1: same-model self-stitch / block bypass.
# Usage: sbatch launch_self_stitch.sbatch <model> <weights> <dataset> <intervals> <out> [seed]
#SBATCH --job-name=stitch
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=runs/stitch_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
"$PYTHON" scripts/run_self_stitch.py \
  --model "$1" --weights "$2" --dataset "$3" \
  --root "/projects/EEG-foundation-model/yinghao/FMCA-AV/$3" \
  --samples 8000 --intervals "$4" --coordinate-budget 64 --gram-bound 0.05 --seed "${6:-1}" --out "$5"
