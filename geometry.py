
import functools

import nibabel as nib
import numpy as np
from scipy import ndimage

from paths import MNI

# decision: RTOG/NRG phase 1 is the whole abnormality plus 2 cm
MARGIN_MM = 20


def ml_per_voxel(image):
    return float(np.prod(image.header.get_zooms()[:3])) / 1000.0


def distance(mask, image):
    return ndimage.distance_transform_edt(~mask, sampling=image.header.get_zooms()[:3])


@functools.lru_cache(maxsize=1)
def template_brain():
    # explainer: cached and shared, don't write into it
    return np.asarray(nib.load(MNI).dataobj) > 0


def grow(seg_image, brain, mm=MARGIN_MM):
    # decision: clip to the brain, a raw expansion spills into skull and air
    return (distance(np.asarray(seg_image.dataobj) != 0, seg_image) <= mm) & brain


def outline(mask):
    # decision: outline only, a filled 2 cm margin swamps the slice
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner
