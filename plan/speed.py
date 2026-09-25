# Ingredients for the speed field, and the geodesic solve. Everything is MNI.

import functools

import nibabel as nib
import numpy as np
from scipy import ndimage

from yale import atlas, connect, graph
from yale import tracts as tractography
from paths import MNI, out_dir

# why: a thinner tube is erased by deliverability smoothing at sigma 5 [penumbra]
TUBE_MM = 6.0


def reference():
    return nib.load(MNI)


def tumor(subject):
    return nib.load(out_dir(subject) / "seg_in_mni.nii.gz")


@functools.lru_cache(maxsize=4)
def crossing(subject):
    # Every centerline through the mass: the ones that go on to reach a parcel,
    # and the ones that land somewhere the atlas has no name for.
    reach, _, _, _, stray = graph.through(subject)
    lines = {n for hits in reach.values() for n, _ in hits}
    return sorted(lines | {n for n, _ in stray})


def corridors(subject, tube_mm=TUBE_MM):
    # A tube of that radius around every tract the tumor sits on.
    seg = tumor(subject)
    lines = tractography.centerlines()
    inverse = np.linalg.inv(seg.affine)
    shape = np.array(seg.shape)

    spine = np.zeros(seg.shape, dtype=bool)
    for n in crossing(subject):
        ijk = np.rint(nib.affines.apply_affine(inverse, lines[n])).astype(int)
        keep = ijk[np.all((ijk >= 0) & (ijk < shape), axis=1)]
        spine[keep[:, 0], keep[:, 1], keep[:, 2]] = True
    if not spine.any():
        return spine

    far = ndimage.distance_transform_edt(~spine, sampling=seg.header.get_zooms()[:3])
    return far <= tube_mm


@functools.lru_cache(maxsize=4)
def disruption(subject):
    # why: disruption share predicts recurrence location, r = 0.60 [wei23]
    labels, names = atlas.label_volume(reference())
    rows = connect.at_risk(
        atlas.overlap(out_dir(subject) / "seg_in_mni.nii.gz", labels, names))

    share = np.zeros(len(names) + 1, dtype=np.float32)
    seat = {name: i for i, name in enumerate(names, start=1)}
    for row in rows:
        if row["parcel"] in seat:
            share[seat[row["parcel"]]] = row["share"]
    return share[labels]


_LEVEL = {}


def _level_set(seed, dx):
    # Put the zero contour on the seed's own rind, so travel time is measured
    # from the same voxel centers a distance transform measures from. Signing
    # the plain transform instead leaves the contour half a voxel out and
    # overshoots a 2 cm expansion by twice as much.
    key = (seed.shape, np.ravel(dx).tobytes(), hash(seed.tobytes()))
    if key not in _LEVEL:
        step = float(np.min(dx))
        inside = ndimage.distance_transform_edt(seed, sampling=dx)
        outside = ndimage.distance_transform_edt(~seed, sampling=dx)
        # a second solve on the same seed only ever changes the speed field
        _LEVEL.clear()
        _LEVEL[key] = np.where(seed, -(inside - step), outside)
    return _LEVEL[key]


def travel(speed, seed, dx=1.0, narrow=0.0):
    # Travel time from the seed under a variable speed, by fast marching.
    import skfmm

    speed = np.asarray(speed, dtype=np.float64)
    seed = np.ascontiguousarray(np.asarray(seed).astype(bool))
    if not seed.any():
        return np.full(speed.shape, np.inf, dtype=np.float32)
    phi = _level_set(seed, dx)

    # skfmm marches only the unmasked part, but it still builds 1/speed first
    blocked = ~(speed > 0)
    field = skfmm.travel_time(np.ma.MaskedArray(phi, blocked),
                              np.where(blocked, 1.0, speed), dx=dx, narrow=narrow)
    out = np.ma.filled(field, np.inf).astype(np.float32)
    out[~np.isfinite(out)] = np.inf
    return out
