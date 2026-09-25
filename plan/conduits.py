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
