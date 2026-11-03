#!/bin/bash
#SBATCH --job-name=synb0
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/scratch/pi_hpz2/wo88/synb0/logs/%x_%A_%a.out
# explainer: Synb0-DISCO for one staged scan per array task, scan names one per line in scans.txt
# decision: run here not on the laptop, its cpu inference needs about 20 GB
set -euo pipefail
BASE=/nfs/roberts/scratch/pi_hpz2/wo88/synb0
scan=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $BASE/scans.txt)
mkdir -p $BASE/$scan/OUTPUTS
apptainer run -e -B $BASE/$scan/INPUTS:/INPUTS -B $BASE/$scan/OUTPUTS:/OUTPUTS \
    -B $BASE/license.txt:/extra/freesurfer/license.txt $BASE/synb0.sif --stripped
echo "done $scan"
