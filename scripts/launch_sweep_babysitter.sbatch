#!/bin/bash
# The sweep babysitter, as a SLURM CPU job.
#
# It used to run under setsid+nohup on the login node, which survived
# everything except the thing that actually happened: the owning shell
# session ended and took it with it, silently.  Twenty-two minutes of
# missed heartbeats later there were six GPU jobs running and nothing
# left alive to queue their certificates and profiles when they landed.
#
# A CPU job outlives any session, is visible in squeue, and is stopped
# with scancel like everything else here.
#SBATCH --job-name=sweepsit
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=runs/sweepsit_%j.out

set -u
cd /home/infres/yinwang/FMCA-AV
exec ./scripts/babysit_sweep.sh
