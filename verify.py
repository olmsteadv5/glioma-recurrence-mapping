# Check the BraTS preprocessing assumptions before spending compute.

import nibabel as nib
import numpy as np

from paths import MODALITIES, modality, out_dir

SHAPE = (240, 240, 155)
ZOOMS = (1.0, 1.0, 1.0)
BRAIN_ML = (900.0, 2000.0)


def run(subject, directory):
    # Raise if anything BraTS assumes about the input is not true.
    problems, affines = [], {}

    for name in MODALITIES:
        path = modality(directory, subject, name)
        image = nib.load(path)
        data = np.asarray(image.dataobj)

        if image.shape != SHAPE:
            problems.append(f"{name}: shape {image.shape} != {SHAPE}")
        if tuple(round(float(z), 3) for z in image.header.get_zooms()[:3]) != ZOOMS:
            problems.append(f"{name}: not 1 mm isotropic")
        if not np.isfinite(data).all():
            problems.append(f"{name}: NaN or Inf")

        voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0
        brain_ml = float((data > 0).sum()) * voxel_ml
        if not BRAIN_ML[0] <= brain_ml <= BRAIN_ML[1]:
            problems.append(f"{name}: {brain_ml:.0f} mL outside {BRAIN_ML} - skull left on?")

        affines[name] = image.affine

    # same affine everywhere, or the modalities are not co-registered
    reference = affines[MODALITIES[0]]
    for name, affine in affines.items():
        if not np.allclose(affine, reference, atol=1e-4):
            problems.append(f"{name}: affine differs - not co-registered")

    # written before raising, so the panel can say what was wrong afterwards
    out_dir(subject).mkdir(parents=True, exist_ok=True)
    (out_dir(subject) / "verify.txt").write_text("\n".join(problems))
    if problems:
        raise RuntimeError(f"{subject}:\n  " + "\n  ".join(problems))
