# Tumor volumes, and Dice against reference labels where they exist.

import nibabel as nib
import numpy as np
from scipy import ndimage

from paths import LABELS, REFERENCE_SEG, REGIONS, modality, stripped_dir

MARGIN_MM = 20

# old BraTS used 4 for enhancing, 2025 uses 3
LEGACY = {0: 0, 1: 1, 2: 2, 4: 3}


def _to_2025(data):
    present = set(np.unique(data).astype(int).tolist())
    if 4 in present and 3 not in present:
        out = np.zeros_like(data)
        for source, target in LEGACY.items():
            out[data == source] = target
        return out
    return data


def _dice(a, b):
    total = a.sum() + b.sum()
    return None if total == 0 else float(2.0 * np.logical_and(a, b).sum() / total)


def run(subject, seg_path):
    # Volumes in mL, and Dice if this subject has reference labels.
    image = nib.load(seg_path)
    data = np.asarray(image.dataobj).astype(np.int16)
    voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0

    result = {
        "labels_ml": {name: round(float((data == value).sum()) * voxel_ml, 2)
                      for value, name in LABELS.items()
                      if (data == value).any()},
        "regions_ml": {name: round(float(np.isin(data, values).sum()) * voxel_ml, 2)
                       for name, values in REGIONS.items()},
    }

    # the planning margin, clipped to the brain: a fifth of a raw 2 cm
    # expansion falls outside the head, where it means nothing
    brain_path = modality(stripped_dir(subject), subject, "T1")
    if brain_path.exists():
        brain = np.asarray(nib.load(brain_path).dataobj) > 0
        if brain.shape == data.shape:
            far = ndimage.distance_transform_edt(data == 0, sampling=image.header.get_zooms()[:3])
            result["margin_ml"] = round(float(((far <= MARGIN_MM) & brain).sum()) * voxel_ml, 2)

    reference_path = REFERENCE_SEG / f"{subject}_seg.nii.gz"
    if reference_path.exists():
        reference = _to_2025(np.asarray(nib.load(reference_path).dataobj).astype(np.int16))
        if reference.shape == data.shape:
            result["dice"] = {
                name: (None if (d := _dice(np.isin(data, v), np.isin(reference, v))) is None
                       else round(d, 4))
                for name, v in REGIONS.items()}
    return result
