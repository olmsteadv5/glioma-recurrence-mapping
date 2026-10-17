
import nibabel as nib
import numpy as np

from geometry import MARGIN_MM, grow, ml_per_voxel
from paths import LABELS, REFERENCE_SEG, REGIONS, out_dir

# explainer: old BraTS used 4 for enhancing, 2025 uses 3
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
    image = nib.load(seg_path)
    data = np.asarray(image.dataobj).astype(np.int16)
    voxel_ml = ml_per_voxel(image)

    result = {
        "labels_ml": {name: round(float((data == value).sum()) * voxel_ml, 2)
                      for value, name in LABELS.items()
                      if (data == value).any()},
        "regions_ml": {name: round(float(np.isin(data, values).sum()) * voxel_ml, 2)
                       for name, values in REGIONS.items()},
    }

    # decision: SynthStrip's mask not T1 > 0, which drops dark CSF inside the brain
    brain_path = out_dir(subject) / "brainmask.nii.gz"
    if brain_path.exists():
        brain = np.asarray(nib.load(brain_path).dataobj) > 0
        if brain.shape == data.shape:
            grown = grow(image, brain, MARGIN_MM)
            result["margin_ml"] = round(float(grown.sum()) * voxel_ml, 2)

    reference_path = REFERENCE_SEG / f"{subject}_seg.nii.gz"
    if reference_path.exists():
        reference = _to_2025(np.asarray(nib.load(reference_path).dataobj).astype(np.int16))
        if reference.shape == data.shape:
            result["dice"] = {
                name: (None if (d := _dice(np.isin(data, v), np.isin(reference, v))) is None
                       else round(d, 4))
                for name, v in REGIONS.items()}
    return result
