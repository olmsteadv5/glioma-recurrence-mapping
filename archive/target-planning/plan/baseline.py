# The targets a connectome-guided one has to beat, all in MNI.

import functools

import nibabel as nib
import numpy as np
from scipy import ndimage

from paths import LABELS, MNI
from plan import barriers, march
from plan.measure import MARGIN_MM, _ml, _seg

# why: ESTRO-EANO gives 15 mm from enhancement plus cavity, edema excluded [estro23]
ESTRO_MM = 15

# why: ESTRO-EANO keeps 5 mm of the margin at the ventricles [estro23]
VENTRICLE_MM = 5

SNFH = next(v for v, name in LABELS.items() if name == "SNFH")

# how much faster the FLAIR comparator runs inside the edema. A comparator knob,
# not a claim about biology; nothing in the literature sets it.
FLAIR_GAIN = 5.0


@functools.lru_cache(maxsize=32)
def _reach(subject):
    # distance from the tumor, and the template brain, computed once per subject
    data, image = _seg(subject)
    far = ndimage.distance_transform_edt(data == 0,
                                         sampling=image.header.get_zooms()[:3])
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


def gtv(subject):
    # why: the ESTRO margin is measured from enhancing tumor plus cavity, with
    #      edema excluded; the necrotic core sits inside the rim [estro23]
    data, _ = _seg(subject)
    core = np.isin(data, [v for v, name in LABELS.items() if name != "SNFH"])
    return core if core.any() else data > 0


def isotropic(subject, volume_ml, tol=0.01):
    # the volume-matched comparator: the uniform expansion that costs what ours costs
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


def _match(time, room, ml, volume_ml, tol):
    # the same bisection isotropic() uses, on travel time instead of millimeters
    finite = time[room & np.isfinite(time)]
    lo, hi = 0.0, float(finite.max()) if finite.size else 0.0
    mid = hi
    for _ in range(40):
        mid = (lo + hi) / 2
        got = float(((time <= mid) & room).sum()) * ml
        if abs(got - volume_ml) <= tol * volume_ml:
            break
        lo, hi = (mid, hi) if got < volume_ml else (lo, mid)
    return (time <= mid) & room, round(mid, 2)


def estro(subject, mm=ESTRO_MM, ventricle_mm=VENTRICLE_MM):
    # A Euclidean ball cannot say "stops at the falx". The falx is one voxel
    # thick, so subtracting it leaves the margin sitting on the far side anyway.
    # Zeroing the speed on the sheet stops the front instead, which is the rule.
    # There is no optic chiasm mask in this pipeline, so that one reduction of
    # the five is simply not applied, and the target is larger for it.
    m = barriers.masks(subject)
    seed = gtv(subject)
    room = m["brain"] | seed

    speed = np.where(room, 1.0, 0.0).astype(np.float32)
    # why: ESTRO-EANO reduces the margin to 0 mm at these structures [estro23]
    speed[m["falx"] | m["tentorium"] | m["brainstem"]] = 0.0
    speed[m["ventricle"]] = 0.0
    # SynthSeg labels inside a mass are unvalidated, so do not let one block the
    # front before it has left the tumor
    speed[seed] = 1.0

    time = march.travel(speed, seed, narrow=float(mm))
    tissue = (time <= mm) & room

    # the ventricle is a wall the front never crosses, and then the guideline's
    # reduced margin is handed back: 5 mm of lining, no traversal. Capped at the
    # full margin too, so the give-back can never reach past where 15 mm would.
    zooms = _seg(subject)[1].header.get_zooms()[:3]
    near = ndimage.distance_transform_edt(~tissue, sampling=zooms) <= ventricle_mm
    out = ndimage.distance_transform_edt(~seed, sampling=zooms) <= mm
    return tissue | (m["ventricle"] & near & out)


def flair(subject, volume_ml, gain=FLAIR_GAIN, tol=0.01):
    # The cheap patient-specific competitor: grow from the GTV but run fast
    # through the edema, so the budget is spent on tissue that is already
    # abnormal rather than on a ball.
    # why: >80% of non-enhancing samples contain tumor on histopathology [infiltrate17]
    data, image = _seg(subject)
    seed = gtv(subject)
    # the template brain, not SynthSeg, so this baseline needs no tissue stage
    brain = np.asarray(nib.load(MNI).dataobj) > 0
    room = brain | seed

    speed = np.where(room, 1.0, 0.0).astype(np.float32)
    speed[(data == SNFH) & room] = gain

    time = march.travel(speed, seed)
    return _match(time, room, _ml(image), volume_ml, tol)
