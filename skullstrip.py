# Skull-strip with SynthStrip; the mirror ships unstripped scans.

import subprocess

import nibabel as nib
import numpy as np

from paths import (MODALITIES, SYNTHSTRIP, THREADS, modality, out_dir, raw_dir,
                   stripped_dir)

MAX_BRAIN_ML = 2000.0


def _volume_ml(path):
    image = nib.load(path)
    voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0
    return float((np.asarray(image.dataobj) > 0).sum()) * voxel_ml


def run(subject):
    # Mask from the T1, applied to all four modalities so they agree.
    source, dest = raw_dir(subject), stripped_dir(subject)
    dest.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir(subject) / "brainmask.nii.gz"

    result = subprocess.run(
        [str(SYNTHSTRIP), "-i", str(modality(source, subject, "T1")),
         "-m", str(mask_path), "-t", THREADS],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"synthstrip failed: {result.stderr.strip()[:200]}")

    mask = np.asarray(nib.load(mask_path).dataobj) > 0
    for name in MODALITIES:
        image = nib.load(modality(source, subject, name))
        out = np.where(mask, np.asarray(image.dataobj), 0).astype(image.get_data_dtype())
        result_image = nib.Nifti1Image(out, image.affine, image.header)
        result_image.set_data_dtype(image.get_data_dtype())
        nib.save(result_image, modality(dest, subject, name))

    brain_ml = _volume_ml(modality(dest, subject, "T1"))
    if brain_ml > MAX_BRAIN_ML:
        raise RuntimeError(f"{subject}: {brain_ml:.0f} mL after stripping, still has skull")
    return dest
