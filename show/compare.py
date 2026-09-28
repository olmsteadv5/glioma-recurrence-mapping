# Two timepoints of one patient, against each other in MNI, the frame they share.

import functools

import nibabel as nib
import numpy as np

import subprocess

from geometry import ml_per_voxel
from paths import THREADS, modality, out_dir, stripped_dir

PARTS = ("both", "new", "gone")


def _mask(subject):
    image = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")
    return np.asarray(image.dataobj) > 0, image


@functools.lru_cache(maxsize=8)
def _split(first, second):
    a, image = _mask(first)
    b, _ = _mask(second)
    return dict(zip(PARTS, (a & b, b & ~a, a & ~b))), a, b, image


@functools.lru_cache(maxsize=8)
def against(first, second):
    # first is the earlier scan, second the later one
    parts, a, b, image = _split(first, second)
    ml = ml_per_voxel(image)
    shared = float(parts["both"].sum())
    out = {"dice": round(2 * shared / max(float(a.sum() + b.sum()), 1), 3),
           "inside": round(shared / max(float(b.sum()), 1), 3),
           "kept": round(shared / max(float(a.sum()), 1), 3)}
    out.update({name: round(float(mask.sum()) * ml, 2) for name, mask in parts.items()})
    return out


def masks(first, second):
    return _split(first, second)[0]


@functools.lru_cache(maxsize=4)
def surfaces(first, second):
    from show import view
    parts, _, _, image = _split(first, second)
    out = {}
    for name, mask in parts.items():
        made = view._tumor_surface(mask.astype(np.int16), image.affine, 1)
        if made:
            out[name] = made
    return out


@functools.lru_cache(maxsize=8)
def parcels_in(first, second, which):
    # Which parcels each piece of the difference actually touches.
    import nibabel as nib
    from yale.atlas import label_volume
    from paths import MNI

    labels, names = label_volume(nib.load(MNI))
    parts = _split(first, second)[0]
    out = set()
    for name in which:
        for value in np.unique(labels[parts[name]]):
            if value > 0:
                out.add(names[int(value) - 1])
    return frozenset(out)


def _inverse(subject):
    # the way back from MNI, computed once and kept
    warp = out_dir(subject) / "warp.nii.gz"
    back = out_dir(subject) / "warp_inverse.nii.gz"
    if not back.exists():
        subprocess.run(["greedy", "-d", "3", "-iw", str(warp), str(back),
                        "-threads", THREADS], capture_output=True)
    return back


def native(subject, first, second):
    # The three pieces in this subject's own space, not the atlas's.
    out, reference = {}, modality(stripped_dir(subject), subject, "T1")
    if not reference.exists():
        return {}
    back = _inverse(subject)
    source = nib.load(out_dir(first) / "seg_in_mni.nii.gz")
    for name, mask in masks(first, second).items():
        dest = out_dir(subject) / f"compare_{name}.nii.gz"
        if not dest.exists():
            held = out_dir(subject) / f"compare_{name}_mni.nii.gz"
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), source.affine), held)
            subprocess.run(
                ["greedy", "-d", "3", "-rf", str(reference), "-ri", "LABEL", "0.2vox",
                 "-rm", str(held), str(dest), "-r",
                 str(out_dir(subject) / "affine.mat") + ",-1", str(back),
                 "-threads", THREADS], capture_output=True)
            held.unlink(missing_ok=True)
        if dest.exists():
            out[name] = np.asarray(nib.load(dest).dataobj) > 0
    return out
