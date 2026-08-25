#!/bin/bash
# E-D2: interface-count scan at fixed endpoints.
# Usage: sbatch launch_length_scan.sbatch <model> <weights> <dataset> <out> [seed]
#SBATCH --job-name=lscan
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=runs/lscan_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
"$PYTHON" scripts/run_length_scan.py \
  --model "$1" --weights "$2" --dataset "$3" \
  --root "/projects/EEG-foundation-model/yinghao/FMCA-AV/$3" \
  --samples 8000 --lengths 1,2,4,8,14 --seed "${5:-1}" --out "$4"
