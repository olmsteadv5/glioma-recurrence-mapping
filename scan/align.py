# decision: rigid on purpose, it never scales so volumes stay as scanned (an MNI warp doesn't)

import nibabel as nib
import numpy as np

from paths import THREADS, anatomy, out_dir, pair_dir, stripped_dir
from scan.register import _greedy, _metric_mask


def first_of(later):
    return later[:-2] + "11"


def _brain_dice(a, b):
    a = np.asarray(nib.load(a).dataobj) > 0
    b = np.asarray(nib.load(b).dataobj) > 0
    return 2 * float((a & b).sum()) / float(a.sum() + b.sum())


def run(later):
    first = first_of(later)
    destination = pair_dir(later)
    fixed = anatomy(stripped_dir(later), later)
    moving = anatomy(stripped_dir(first), first)
    log = destination / "align.log"
    log.write_text("")
    rigid = destination / "rigid_first_to_later.mat"
    mask = destination / "align_mask.nii.gz"
    seg = destination / "first_seg_in_later.nii.gz"

    # decision: follow-up brain minus tumor drives the fit. -gm masks fixed, -mm crashes in affine mode
    _metric_mask(fixed, out_dir(later) / "seg.nii.gz", mask)
    # decision: NCC not NMI, NMI drifted far off a good starting point
    _greedy(["-d", 3, "-a", "-dof", 6, "-m", "NCC", "4x4x4", "-i", fixed, moving, "-gm", mask,
             "-ia-identity", "-o", rigid, "-n", "100x100x50", "-threads", THREADS], log)
    _greedy(["-d", 3, "-rf", fixed, "-rm", moving, destination / "first_t1_in_later.nii.gz",
             "-ri", "LABEL", "0.2vox", "-rm", out_dir(first) / "seg.nii.gz", seg,
             "-r", rigid, "-threads", THREADS], log)
    # decision: reject a fit clearly worse than the raw grid, small drops are just noise
    before = _brain_dice(fixed, moving)
    after = _brain_dice(fixed, destination / "first_t1_in_later.nii.gz")
    with open(log, "a") as fh:
        fh.write(f"\nbrain dice {before:.4f} as shipped, {after:.4f} aligned\n")
    if after < before - 0.02:
        raise RuntimeError(f"{later}: alignment made the fit worse, see {log}")
    return seg
