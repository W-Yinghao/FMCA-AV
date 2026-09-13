#!/bin/bash
# Train one external SSL baseline from a config.
# Usage: sbatch launch_baseline_train.sbatch <config> <seed> <run-name>
#SBATCH --job-name=sslmatch
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=runs/sslmatch_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
ROOT=/projects/EEG-foundation-model/yinghao/FMCA-AV/experiment_runs
"$PYTHON" fmca_av/baseline_cli.py train \
  --config "$1" --seed "$2" --output "$ROOT/$3"
