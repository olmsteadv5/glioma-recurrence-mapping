# The targets a connectome-guided one has to beat, all in MNI.

import functools

import nibabel as nib
import numpy as np
from scipy import ndimage

from paths import MNI, out_dir
from plan.measure import MARGIN_MM, _ml, _seg

# why: ESTRO-EANO gives 15 mm from enhancement plus cavity, edema excluded [estro23]
ESTRO_MM = 15


@functools.lru_cache(maxsize=32)
def _reach(subject):
    # distance from the tumor, and the template brain, computed once per subject
    data, image = _seg(subject)
    far = ndimage.distance_transform_edt(data == 0, sampling=image.header.get_zooms()[:3])
    return far, np.asarray(nib.load(MNI).dataobj) > 0, image


def expand(subject, mm):
    far, brain, _ = _reach(subject)
    return (far <= mm) & brain


def volume(subject, mask):
    return round(float(mask.sum()) * _ml(_seg(subject)[1]), 2)


def rtog(subject):
    # the 2 cm expansion, grown from the whole segmentation including edema, so
    # this is the phase-1 volume rather than the 60 Gy one
    return expand(subject, MARGIN_MM)


def isotropic(subject, volume_ml, tol=0.01):
    # the volume-matched comparator: the sphere that costs what our target costs
    far, brain, image = _reach(subject)
    ml = _ml(image)
    lo, hi = 0.0, float(far[brain].max())
    for _ in range(40):
        mid = (lo + hi) / 2
        got = float(((far <= mid) & brain).sum()) * ml
        if abs(got - volume_ml) <= tol * volume_ml:
            break
        lo, hi = (mid, hi) if got < volume_ml else (lo, mid)
    return (far <= mid) & brain, round(mid, 2)


def estro(subject):
    # 15 mm with the guideline's barrier reductions. The reductions need the
    # tissue segmentation, so until that stage lands this is the margin alone.
    raise NotImplementedError(
        "estro() needs falx, tentorium and ventricle masks from plan/barriers.py")
