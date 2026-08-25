#!/bin/bash
# Pilot A3: deterministic pretrained depth chain.
# Usage: sbatch launch_depth_chain.sbatch <model> <weights> <dataset> <out> [seed] [floor] [budget]
#SBATCH --job-name=depthchain
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=runs/depthchain_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
"$PYTHON" scripts/run_depth_chain.py \
  --model "$1" --weights "$2" --dataset "$3" \
  --root "/projects/EEG-foundation-model/yinghao/FMCA-AV/$3" \
  --samples 8000 --coordinate-budget "${7:-64}" --variance-floor "${6:-0.01}" --seed "${5:-1}" --out "$4"
