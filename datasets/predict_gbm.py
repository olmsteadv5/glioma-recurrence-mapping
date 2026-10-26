# explainer: PREDICT-GBM (TUM, LUMIERE, RHUH) ships stripped, segmented SRI24 scans, with the follow-up already warped onto the pre-op grid

import nibabel as nib
import numpy as np

from paths import COLLECTIONS, PREDICT, RHUH, THREADS
from scan.register import _greedy

SOURCE = PREDICT / "extracted" / "predict_gbm"
RHUH_NII = RHUH / "download" / "PKG - RHUH-GBM-nii-v1" / "RHUH-GBM_nii_v1"
# decision: the SRI24 grid every other scan here uses
SRI = np.array([[-1, 0, 0, 0], [0, -1, 0, 239], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
# decision: TUM and LUMIERE headers say x and y run the other way, so flip both onto SRI24 like the header says
FLIPPED = ("T", "L")
# explainer: the follow-up needs no registration of its own
FOLLOWUP_ON_PREOP_GRID = True


def _kind(number):
    return COLLECTIONS[number // 10000][8]


def folder(number):
    n = number % 10000
    return SOURCE / {"T": f"tum_gbm_{n:03d}", "L": f"Patient-{n:03d}", "R": f"RHUH-{n:04d}"}[_kind(number)]


def number_of(name):
    # explainer: tum_gbm_002 -> 50002, Patient-001 -> 60001, RHUH-0001 -> 70001
    for base, prefix in ((50000, "tum_gbm_"), (60000, "Patient-"), (70000, "RHUH-")):
        if name.startswith(prefix):
            return base + int(name[len(prefix):])
    return None


def everyone():
    return sorted(n for n in (number_of(p.name) for p in SOURCE.iterdir()) if n is not None)


def scans(number, timepoint):
    stems = ({"T1GD": "t1c_bet_normalized", "FLAIR": "flair_bet_normalized"} if timepoint == 1
             else {"T1GD": "t1c_warped_longitudinal"})
    return {name: folder(number) / f"{stem}.nii.gz" for name, stem in stems.items()}


def brainmask(number):
    # decision: the follow-up is on the pre-op grid, so both share the pre-op mask
    return folder(number) / "t1c_bet_mask.nii.gz"


def labels(number, timepoint):
    return folder(number) / ("tumor_seg.nii.gz" if timepoint == 1 else "recurrence_preop.nii.gz")


def load(path, number, is_labels=False):
    image = nib.load(path)
    data = np.asarray(image.dataobj)
    if _kind(number) in FLIPPED:
        data = data[::-1, ::-1, :]
    data = np.rint(data).astype(np.int16) if is_labels else data.astype(np.float32)
    out = nib.Nifti1Image(np.ascontiguousarray(data), SRI)
    out.set_data_dtype(np.int16 if is_labels else np.float32)
    return out


def expert(number, timepoint, fixed, dest, log):
    # decision: RHUH's own expert labels over PREDICT's automatic ones. affine for pre-op, + deformable for the recurrence
    if _kind(number) != "R":
        return False
    n = number % 10000
    case = RHUH_NII / f"RHUH-{n:04d}" / ("0" if timepoint == 1 else "2")
    found = sorted(case.glob("*segmentation*.nii.gz"))
    scan = case / f"RHUH-{n:04d}_{0 if timepoint == 1 else 2}_t1ce.nii.gz"
    if not found or not scan.exists():
        return False
    matrix = dest.parent / "expert_affine.mat"
    _greedy(["-d", 3, "-a", "-dof", 12, "-m", "NMI", "-i", fixed, scan, "-o", matrix,
             "-ia-image-centers", "-n", "100x50x20", "-threads", THREADS], log)
    chain = [matrix]
    if timepoint == 2:
        warp = dest.parent / "expert_warp.nii.gz"
        _greedy(["-d", 3, "-m", "NCC", "2x2x2", "-i", fixed, scan, "-it", matrix, "-o", warp,
                 "-n", "60x30x10", "-threads", THREADS], log)
        chain = [warp, matrix]
    _greedy(["-d", 3, "-rf", fixed, "-ri", "LABEL", "0.2vox", "-rm", found[0], dest, "-r", *chain,
             "-threads", THREADS], log)
    return True
