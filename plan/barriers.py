# Dural barriers and the ventricle rind, read off the SynthSeg labels, in MNI.

import functools

import nibabel as nib
import numpy as np
from scipy import ndimage

from paths import MNI, out_dir
from scan.tissue import (BRAINSTEM, CEREBELLUM, CORTEX, CSF, HIPPOCAMPUS, LEFT,
                         MIDLINE_VENTRICLE, RIGHT, VENTRICLE, WM, to_mni)
from yale import atlas

# why: the callosum is exactly how tumor crosses the midline [butterfly22]
CALLOSUM = ("L_CC_A", "L_CC_B", "L_CC_C", "R_CC_A", "R_CC_B", "R_CC_C")

# How far from either side a sheet is still allowed to run. SynthSeg does not
# label dura, so an interface has to be found in the CSF gap instead of on a
# shared boundary. Measured on 00055_11 the tentorial gap reaches 10 mm from
# each side; at 8 mm both sheets have holes and the front walks straight
# through, at 12 mm neither does.
GAP_MM = 12.0

# why: the SVZ is contoured as a 5 mm expansion of the lateral ventricle [svzdef20]
SVZ_MM = 5.0

_BOX = ndimage.generate_binary_structure(3, 3)


def labels(subject):
    return np.asarray(nib.load(to_mni(subject)).dataobj).astype(np.int16)


def _group(tissue, values):
    return np.isin(tissue, values)


def hemispheres(tissue):
    # LEFT and RIGHT carry the cerebellum, and the falx is not its barrier.
    left = [v for v in LEFT if v not in CEREBELLUM]
    right = [v for v in RIGHT if v not in CEREBELLUM]
    return _group(tissue, left), _group(tissue, right)


def _interface(near, far, gap=GAP_MM):
    # The watershed between two regions that do not touch: every voxel closer to
    # one than the other is on that side, and the sheet is the wall between the
    # two sides, taken from both sides so no path slips across a corner. Clipped
    # to the gap, or it would run off through the whole volume.
    inner = ndimage.distance_transform_edt(~near)
    outer = ndimage.distance_transform_edt(~far)
    side = inner < outer
    wall = ((ndimage.binary_dilation(side, _BOX) & ~side)
            | (ndimage.binary_dilation(~side, _BOX) & side))
    return wall & (inner <= gap) & (outer <= gap)


@functools.lru_cache(maxsize=1)
def _callosum():
    # label_volume ignores its argument once the resampled atlas is cached, and
    # returns MNI labels either way.
    parcels, names = atlas.label_volume(nib.load(MNI))
    seat = {name: i for i, name in enumerate(names, start=1)}
    return _group(parcels, [seat[n] for n in CALLOSUM])


def falx(tissue):
    left, right = hemispheres(tissue)
    # why: ESTRO-EANO reduces the margin to 0 mm at the falx [estro23]
    return _interface(left, right) & ~_callosum()


def tentorium(tissue):
    left, right = hemispheres(tissue)
    # why: ESTRO-EANO reduces the margin to 0 mm at the tentorium [estro23]
    return _interface(left | right, _group(tissue, CEREBELLUM + BRAINSTEM))


def svz(tissue):
    # The niche is the lining, so the ventricle and its CSF are not in it. Each
    # rind is held to its own hemisphere: the septum is thin enough that 5 mm
    # off the left ventricle otherwise reaches the right one.
    fluid = _group(tissue, VENTRICLE + MIDLINE_VENTRICLE + CSF)
    solid = (tissue > 0) & ~fluid
    rinds = []
    for side in (LEFT, RIGHT):
        lateral = _group(tissue, [v for v in VENTRICLE if v in side])
        near = ndimage.distance_transform_edt(~lateral) <= SVZ_MM
        rinds.append(near & solid & _group(tissue, side))
    return rinds[0], rinds[1]


def svz_contact(subject):
    # why: dosing the SVZ helps on contact and costs cognition otherwise [svz25]
    tumor = np.asarray(
        nib.load(out_dir(subject) / "seg_in_mni.nii.gz").dataobj) > 0
    left, right = svz(labels(subject))
    return bool((tumor & left).any()), bool((tumor & right).any())


@functools.lru_cache(maxsize=4)
def masks(subject):
    tissue = labels(subject)
    return {
        # the template's own brain, not SynthSeg's. Every comparator has to run
        # on the same ground or part of any difference is just a bigger field.
        "brain": np.asarray(nib.load(MNI).dataobj) > 0,
        "wm": _group(tissue, WM),
        "cortex": _group(tissue, CORTEX),
        "csf": _group(tissue, CSF),
        "ventricle": _group(tissue, VENTRICLE + MIDLINE_VENTRICLE),
        "cerebellum": _group(tissue, CEREBELLUM),
        "brainstem": _group(tissue, BRAINSTEM),
        "hippocampus": _group(tissue, HIPPOCAMPUS),
        "falx": falx(tissue),
        "tentorium": tentorium(tissue),
    }
