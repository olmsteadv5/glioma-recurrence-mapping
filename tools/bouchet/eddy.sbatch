#!/bin/bash
#SBATCH --job-name=eddy
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/scratch/pi_hpz2/wo88/eddy/logs/%x_%A_%a.out
# explainer: FSL eddy on a gpu for one staged scan per array task, scan names one per line in scans.txt
# decision: same options as the laptop run so the gpu and cpu results compare like for like
set -euo pipefail
BASE=/nfs/roberts/scratch/pi_hpz2/wo88/eddy
scan=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $BASE/scans.txt)
d=$BASE/$scan
module load FSL/6.0.7.22
# decision: the module leaves the output format unset and eddy refuses to write images without it
export FSLOUTPUTTYPE=NIFTI_GZ
eddy_cuda --imain=$d/unrung.nii.gz --mask=$d/brain_mask.nii.gz --acqp=$d/acqp.txt --index=$d/index.txt \
    --bvecs=$d/dwi.bvec --bvals=$d/dwi.bval --topup=$d/topup --out=$d/eddy \
    --repol --slm=linear --fwhm=10,0,0,0,0 --cnr_maps
echo "done $scan"
