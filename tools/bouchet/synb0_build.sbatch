#!/bin/bash
#SBATCH --job-name=synb0_build
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/scratch/pi_hpz2/wo88/synb0/logs/%x_%j.out
# explainer: one-time build of the Synb0-DISCO container, caches kept on scratch since home is small
set -euo pipefail
BASE=/nfs/roberts/scratch/pi_hpz2/wo88/synb0
export APPTAINER_CACHEDIR=$BASE/cache APPTAINER_TMPDIR=$BASE/tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
apptainer build $BASE/synb0.sif docker://leonyichencai/synb0-disco:v3.1
