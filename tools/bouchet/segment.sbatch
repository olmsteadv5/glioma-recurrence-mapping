#!/bin/bash
#SBATCH --job-name=brats25_1
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/scratch/pi_hpz2/wo88/brats/logs/%x_%j.out

# explainer: the BraTS 2025 first-place model on the scans named on the command line, every scan in inputs/ when none are named
set -euo pipefail

BRATS=/nfs/roberts/scratch/pi_hpz2/wo88/brats
SIF=$BRATS/aia-bots.sif
# decision: stage on the gpu node's local disk, scratch is a slow network filesystem
WORK=/tmp/wo88_$SLURM_JOB_ID
mkdir -p $WORK/input $WORK/output $WORK/tmp

scans=("$@")
if [ ${#scans[@]} -eq 0 ]; then
    scans=($(ls $BRATS/inputs))
fi

# explainer: the model wants BraTS-style case folders, named the way the brats package names them, patient number then timepoint
for scan in "${scans[@]}"; do
    number=${scan:10:5}; timepoint=${scan:16:1}
    case=BraTS-GLI-$(printf %05d $((10#$number * 10 + timepoint)))-100
    src=$BRATS/inputs/$scan/stripped
    mkdir -p $WORK/input/$case
    cp $src/${scan}_T1GD.nii.gz  $WORK/input/$case/$case-t1c.nii.gz
    cp $src/${scan}_T1.nii.gz    $WORK/input/$case/$case-t1n.nii.gz
    cp $src/${scan}_FLAIR.nii.gz $WORK/input/$case/$case-t2f.nii.gz
    cp $src/${scan}_T2.nii.gz    $WORK/input/$case/$case-t2w.nii.gz
    echo "$case $scan" >> $WORK/names.txt
done
echo "staged ${#scans[@]} scans on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# explainer: --nv hands it the gpu, the model runs from /workspace and keeps intermediates in /workspace/tmp
apptainer run --nv --cwd /workspace \
    -B $WORK/input:/input -B $WORK/output:/output -B $WORK/tmp:/workspace/tmp \
    $SIF

# explainer: each result goes back under its real scan name, then the local disk is cleared
while read case scan; do
    mkdir -p $BRATS/outputs/$scan
    # explainer: matched by its case id, the way the brats package finds it
    result=$(ls $WORK/output/*${case#BraTS-GLI-}*.nii.gz 2>/dev/null | head -1 || true)
    if [ -n "$result" ]; then
        cp "$result" $BRATS/outputs/$scan/seg.nii.gz && echo "done $scan"
    else
        echo "MISSING $scan"
    fi
done < $WORK/names.txt
rm -rf $WORK
