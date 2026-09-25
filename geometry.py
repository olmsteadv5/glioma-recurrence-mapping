# Growing and outlining masks. Shared, because two packages were doing it twice.

import numpy as np
from scipy import ndimage

# why: RTOG grows the phase-1 volume 2 cm from the whole abnormality [recur21]
MARGIN_MM = 20


def grow(seg_image, brain, mm=MARGIN_MM):
    # the planning margin, clipped to the brain because a fifth of a raw
    # expansion lands in skull and air
    far = ndimage.distance_transform_edt(np.asarray(seg_image.dataobj) == 0,
                                         sampling=seg_image.header.get_zooms()[:3])
    return (far <= mm) & brain


def outline(mask):
    # the boundary only, so a 2 cm envelope does not swamp the slice
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner
