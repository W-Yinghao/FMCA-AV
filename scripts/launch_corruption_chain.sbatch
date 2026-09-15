#!/bin/bash
# Corruption-path pilot on ImageNet-C.
# Usage: sbatch launch_corruption_chain.sbatch <corruption> <out> [seed] [severities] [model] [weights]
#SBATCH --job-name=corrchain
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=runs/corrchain_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
"$PYTHON" scripts/run_corruption_chain.py \
  --corruption "$1" --seed "${3:-1}" --severities "${4:-1,3,5}" \
  --model "${5:-resnet50}" --weights "${6:-IMAGENET1K_V2}" --out "$2"
