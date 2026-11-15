#!/bin/bash
#SBATCH --job-name=brats_build
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/scratch/pi_hpz2/wo88/brats/logs/%x_%j.out
# explainer: builds the BraTS 2025 first-place segmenter image that segment.sbatch runs, scratch purges it after a few weeks
set -euo pipefail
BRATS=/nfs/roberts/scratch/pi_hpz2/wo88/brats
export APPTAINER_CACHEDIR=/nfs/roberts/scratch/pi_hpz2/wo88/apptainer_cache
export APPTAINER_TMPDIR=$APPTAINER_CACHEDIR/tmp
mkdir -p $APPTAINER_TMPDIR
apptainer build $BRATS/aia-bots.sif docker://brainles/brats25_gli_aia-bots:latest
ls -la $BRATS/aia-bots.sif
