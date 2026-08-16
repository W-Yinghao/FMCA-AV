#!/bin/bash
# Wave 0 certificate suite (CPU only, preregistered).
# Launch ONLY after explicit user go; prereg is
# prereg/WAVE0_CERTIFICATE_PREREG_FROZEN_20260816.md.
# Units are skip-if-done, so resubmission after any kill wastes nothing.
#SBATCH --job-name=wave0-certificate
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=runs/wave0_certificate_%j.out

set -euo pipefail
cd /home/infres/yinwang/FMCA-AV
PYTHON=/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python

"$PYTHON" scripts/run_wave0_certificate_suite.py --group all \
  --output-root results/wave0/20260816_path_supported_certificate_v1

# Aggregation runs separately after coverage is complete:
#   "$PYTHON" scripts/aggregate_wave0_certificate.py \
#     --output-root results/wave0/20260816_path_supported_certificate_v1
