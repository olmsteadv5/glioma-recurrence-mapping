# Rigidly align the same patient's pre-op scan onto a follow-up, so the pair can be
# compared in patient space, on the scan the tumor came back on. Rigid moves and
# turns but never scales, so every volume stays what the scan says, which a warp
# to MNI cannot promise.

import nibabel as nib
import numpy as np

from paths import THREADS, modality, out_dir, pair_dir, stripped_dir
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
    fixed = modality(stripped_dir(later), later, "T1")
    moving = modality(stripped_dir(first), first, "T1")
    log = destination / "align.log"
    log.write_text("")
    rigid = destination / "rigid_first_to_later.mat"
    mask = destination / "align_mask.nii.gz"
    seg = destination / "first_seg_in_later.nii.gz"

    # the follow-up brain minus its tumor drives the fit, so neither lesion can.
    # -gm masks the fixed image; -mm is the one that crashes in affine mode
    _metric_mask(fixed, out_dir(later) / "seg.nii.gz", mask)
    # both scans ship on the SRI24 grid, so the headers already roughly line up.
    # NCC, since NMI wandered 21 degrees off a good start on 00130
    _greedy(["-d", 3, "-a", "-dof", 6, "-m", "NCC", "4x4x4", "-i", fixed, moving, "-gm", mask,
             "-ia-identity", "-o", rigid, "-n", "100x100x50", "-threads", THREADS], log)
    _greedy(["-d", 3, "-rf", fixed, "-rm", moving, destination / "first_t1_in_later.nii.gz",
             "-ri", "LABEL", "0.2vox", "-rm", out_dir(first) / "seg.nii.gz", seg,
             "-r", rigid, "-threads", THREADS], log)
    # never accept an alignment that fits clearly worse than the shipped grid did.
    # brain outlines move a little between scans anyway, so small drops are noise;
    # the NMI failure on 00130 fell from 0.965 to 0.777
    before = _brain_dice(fixed, moving)
    after = _brain_dice(fixed, destination / "first_t1_in_later.nii.gz")
    with open(log, "a") as fh:
        fh.write(f"\nbrain dice {before:.4f} as shipped, {after:.4f} aligned\n")
    if after < before - 0.02:
        raise RuntimeError(f"{later}: alignment made the fit worse, see {log}")
    return seg
