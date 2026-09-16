# Register SRI24 -> MNI152 (the YBA reference space), carrying the tumor along.

import subprocess

import nibabel as nib
import numpy as np
from scipy import ndimage

from paths import MNI, THREADS, modality, out_dir

DILATE_MM = 5


def _greedy(args, log):
    with open(log, "a") as fh:
        fh.write(f"\n$ greedy {' '.join(str(a) for a in args)}\n")
        fh.flush()
        result = subprocess.run(["greedy"] + [str(a) for a in args],
                                stdout=fh, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"greedy failed ({result.returncode}), see {log}")


def _metric_mask(t1_path, seg_path, dest):
    # Brain minus the tumor, so the lesion cannot drive the alignment.
    image = nib.load(t1_path)
    brain = np.asarray(image.dataobj) > 0
    lesion = np.asarray(nib.load(seg_path).dataobj) > 0
    lesion = ndimage.binary_dilation(lesion, iterations=DILATE_MM)

    mask = (brain & ~lesion).astype(np.uint8)
    out = nib.Nifti1Image(mask, image.affine)
    out.set_data_dtype(np.uint8)
    nib.save(out, dest)
    return float(mask.sum()) / max(int(brain.sum()), 1)


def run(subject, stripped, seg_path):
    # Affine, then deformable with the lesion masked out.
    destination = out_dir(subject)
    t1 = modality(stripped, subject, "T1")
    log = destination / "registration.log"
    log.write_text("")

    affine = destination / "affine.mat"
    warp = destination / "warp.nii.gz"
    mask = destination / "metric_mask.nii.gz"
    seg_mni = destination / "seg_in_mni.nii.gz"

    kept = _metric_mask(t1, seg_path, mask)

    # -mm crashes in affine mode; a global transform cannot warp locally anyway
    _greedy(["-d", 3, "-a", "-m", "NMI", "-i", MNI, t1, "-o", affine,
             "-ia-image-centers", "-n", "100x100x50", "-threads", THREADS], log)

    # -mm only works with NCC. this is the stage mass effect actually breaks.
    # -oinv gets the way back for free, which is what puts MNI work in this
    # subject's own space later
    _greedy(["-d", 3, "-m", "NCC", "2x2x2", "-i", MNI, t1, "-mm", mask,
             "-it", affine, "-o", warp, "-oinv", destination / "warp_inverse.nii.gz",
             "-n", "100x50x20", "-threads", THREADS], log)

    _greedy(["-d", 3, "-rf", MNI, "-rm", t1, destination / "t1_in_mni.nii.gz",
             "-ri", "LABEL", "0.2vox", "-rm", seg_path, seg_mni,
             "-r", warp, affine, "-threads", THREADS], log)

    return seg_mni, kept
