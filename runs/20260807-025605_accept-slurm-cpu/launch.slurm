#!/bin/bash
set -eu
cd -- /home/infres/yinwang/FMCA-AV
exec /home/infres/yinwang/anaconda3/bin/python3 -m harness.runner --run 20260807-025605_accept-slurm-cpu
