#!/bin/bash
# Generic GPU launcher for analysis runners that encode through the network.
#
# Anything that runs a forward pass belongs here, not on the CPU partition:
# the linear algebra in these scripts is 128x128, the cost is the encoder.
# Five convergence sweeps were submitted to --partition=CPU and reached
# N=2500 of 20000 in eight and a half hours for exactly that reason.
#
# Usage: sbatch launch_analysis_gpu.sbatch <python-script> [args...]
#SBATCH --job-name=analysisgpu
#SBATCH --partition=A100,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=runs/analysisgpu_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
"$PYTHON" "$@"
