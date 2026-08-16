#!/bin/bash
# Wave 0 certificate suite (CPU only, preregistered; user go given 2026-08-16).
# Prereg: prereg/WAVE0_CERTIFICATE_PREREG_FROZEN_20260816.md
# Units are skip-if-done, so resubmission after any kill wastes nothing.
# Usage: sbatch launch_wave0_certificate.sbatch <group> [unit-prefix]
#SBATCH --job-name=wave0-cert
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=runs/wave0_certificate_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python
GROUP="${1:-all}"
PREFIX="${2:-}"

if [ -n "$PREFIX" ]; then
  "$PYTHON" scripts/run_wave0_certificate_suite.py --group "$GROUP" \
    --unit-prefix "$PREFIX" \
    --output-root results/wave0/20260816_path_supported_certificate_v1
else
  "$PYTHON" scripts/run_wave0_certificate_suite.py --group "$GROUP" \
    --output-root results/wave0/20260816_path_supported_certificate_v1
fi

# Aggregation runs separately after coverage is complete:
#   "$PYTHON" scripts/aggregate_wave0_certificate.py \
#     --output-root results/wave0/20260816_path_supported_certificate_v1
