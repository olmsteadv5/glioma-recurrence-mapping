# Growing and outlining masks. Shared, because two packages were doing it twice.

import functools

import nibabel as nib
import numpy as np
from scipy import ndimage

from paths import MNI

# why: RTOG grows the phase-1 volume 2 cm from the whole abnormality [recur21]
MARGIN_MM = 20


def ml_per_voxel(image):
    return float(np.prod(image.header.get_zooms()[:3])) / 1000.0


def distance(mask, image):
    # millimeters from every voxel to the nearest voxel of the mask
    return ndimage.distance_transform_edt(~mask, sampling=image.header.get_zooms()[:3])


@functools.lru_cache(maxsize=1)
def template_brain():
    # shared and cached, so callers must not write into it
    return np.asarray(nib.load(MNI).dataobj) > 0


def grow(seg_image, brain, mm=MARGIN_MM):
    # the planning margin, clipped to the brain because a fifth of a raw
    # expansion lands in skull and air
    return (distance(np.asarray(seg_image.dataobj) != 0, seg_image) <= mm) & brain


def outline(mask):
    # the boundary only, so a 2 cm envelope does not swamp the slice
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner
