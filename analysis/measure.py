# explainer: where the recurrence sits relative to the first tumor, in MNI or the follow-up's own space

import csv
import functools

import nibabel as nib
import numpy as np
from scipy import ndimage

from geometry import MARGIN_MM, distance, ml_per_voxel, template_brain
from paths import PARTICIPANTS, REGIONS, SCANS, out_dir, pair_dir

# decision: Lee pattern classes, by share of the recurrence inside the high-dose volume
PATTERNS = ((0.95, "central"), (0.80, "in-field"), (0.20, "marginal"), (0.0, "distant"))


SPACES = ("native", "mni")


def later_of(first):
    return first[:-2] + "21"


def _seg_file(subject, space):
    if space == "mni":
        return out_dir(subject) / "seg_in_mni.nii.gz"
    if subject.endswith("_11"):
        return pair_dir(subject) / "first_seg_in_later.nii.gz"
    return out_dir(subject) / "seg.nii.gz"


@functools.lru_cache(maxsize=16)
def _seg(subject, space):
    image = nib.load(_seg_file(subject, space))
    return np.asarray(image.dataobj).astype(np.int16), image


@functools.lru_cache(maxsize=4)
def _brain(first, space):
    # explainer: cached and shared, don't write into it
    if space == "mni":
        return template_brain()
    return np.asarray(nib.load(out_dir(later_of(first)) / "brainmask.nii.gz").dataobj) > 0


def _in(data, region):
    return np.isin(data, REGIONS[region])


@functools.lru_cache(maxsize=16)
def _far(subject, region, space):
    data, image = _seg(subject, space)
    seed = _in(data, region)
    return distance(seed if seed.any() else data > 0, image)


def mask(subject, region, space):
    data, _ = _seg(subject, space)
    return _in(data, region)


def pattern(share):
    return next(name for floor, name in PATTERNS if share > floor or floor == 0.0)


def overlap(first, later, region, space):
    a, _ = _seg(first, space)
    b, image = _seg(later, space)
    ml = ml_per_voxel(image)
    was, now = _in(a, region), _in(b, region)
    out = {"region": region, "first_ml": round(float(was.sum()) * ml, 2),
           "later_ml": round(float(now.sum()) * ml, 2)}
    if not now.any():
        return out

    shared = float((was & now).sum())
    brain = _brain(first, space)
    # decision: no dose data, so the NRG volumes stand in for dose. pattern classes use the 60 Gy one
    whole = _far(first, "WT", space)[now]
    core = _far(first, "TC", space)[now]
    inside = brain[now]
    high = float(((core <= MARGIN_MM) & inside).mean())
    out.update({"inside": round(shared / float(now.sum()), 4),
                "kept": round(shared / max(float(was.sum()), 1), 4),
                "dice": round(2 * shared / float(was.sum() + now.sum()), 4),
                "new_ml": round(float((now & ~was).sum()) * ml, 2),
                "in_4600": round(float(((whole <= MARGIN_MM) & inside).mean()), 4),
                "in_6000": round(high, 4),
                "pattern": pattern(high),
                "margin_95_mm": round(float(np.percentile(whole, 95)), 1),
                "margin_max_mm": round(float(whole.max()), 1)})
    return out


def reach(first, later, region, space):
    # decision: outside the brain counts as never reached, plans stop at the brain edge
    b, _ = _seg(later, space)
    now = _in(b, region)
    if not now.any():
        return None
    return np.sort(np.where(_brain(first, space)[now], _far(first, "WT", space)[now], np.inf))


@functools.lru_cache(maxsize=1)
def _participants():
    # decision: interval from the dataset's table, TCIA study dates can pick the wrong study
    path = PARTICIPANTS
    if not path.exists():
        return {}
    with open(path) as fh:
        return {r["ID"]: r for r in csv.DictReader(fh, delimiter="\t")}


def _followup(first, column):
    value = _participants().get(later_of(first), {}).get(column)
    return int(value) if value and value.isdigit() else None


def interval(first):
    return _followup(first, "Time_since_baseline_preop")


def pathology(first):
    return _followup(first, "PsP_TP_score")


def voxel_ml(first, space):
    return ml_per_voxel(_seg(first, space)[1])


def margin_ml(first, mm, space):
    _, image = _seg(first, space)
    far = np.sort(_far(first, "WT", space)[_brain(first, space)])
    grown = np.searchsorted(far, mm, side="right") - np.searchsorted(far, 0, side="right")
    return grown * ml_per_voxel(image)


def zones(first, later, region, space):
    # explainer: four buckets: inside the first tumor, within the margin, past it but connected, separate lesions past it
    b, image = _seg(later, space)
    now = _in(b, region)
    if not now.any():
        return None
    far = _far(first, "WT", space)
    lesions, count = ndimage.label(now, structure=np.ones((3, 3, 3)))
    nearest = np.asarray(ndimage.minimum(far, lesions, index=np.arange(1, count + 1)))
    apart = np.isin(lesions, np.flatnonzero(nearest > MARGIN_MM) + 1)
    ml = ml_per_voxel(image)
    return {"inside": float((now & (far == 0)).sum()) * ml,
            "within": float((now & (far > 0) & (far <= MARGIN_MM)).sum()) * ml,
            "beyond": float((now & (far > MARGIN_MM) & ~apart).sum()) * ml,
            "separate": float(apart.sum()) * ml}


def pairs(space, tissue=False):
    # explainer: pass tissue=True when building a target, a missing tissue map would quietly run SynthSeg
    have = {d.name for d in SCANS.iterdir() if _seg_file(d.name, space).exists()
            and (not tissue or not d.name.endswith("_11")
                 or (d / "tissue_in_mni.nii.gz").exists())}
    return [(name, name[:-2] + "21") for name in sorted(have)
            if name.endswith("_11") and name[:-2] + "21" in have]


def review(pairs, space, regions=("WT", "TC", "ET")):
    rows = []
    for first, later in pairs:
        for region in regions:
            rows.append({"patient": first[-8:-3], "days": interval(first),
                         **overlap(first, later, region, space)})
        forget()
    return rows


def forget():
    # explainer: each cached map is a whole volume, clear between patients or memory fills up
    _seg.cache_clear()
    _far.cache_clear()
    _brain.cache_clear()
