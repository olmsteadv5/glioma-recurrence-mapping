# Where the recurrence sits relative to the first tumor, in either space a pair can
# be compared in: MNI, where the atlas lives, or the patient's own follow-up scan,
# which the pre-op scan reaches by a rigid fit that leaves every volume as scanned.

import csv
import functools

import nibabel as nib
import numpy as np
from scipy import ndimage

from geometry import MARGIN_MM, distance, ml_per_voxel, template_brain
from paths import OUTPUTS, REGIONS, out_dir

# why: central >95%, in-field 80-95%, marginal 20-80%, distant <20% of the
#      recurrence inside the high-dose volume [lee99]
PATTERNS = ((0.95, "central"), (0.80, "in-field"), (0.20, "marginal"), (0.0, "distant"))


SPACES = ("native", "mni")


def later_of(first):
    return first[:-2] + "21"


def _seg_file(subject, space):
    if space == "mni":
        return out_dir(subject) / "seg_in_mni.nii.gz"
    if subject.endswith("_11"):
        return out_dir(later_of(subject)) / "first_seg_in_later.nii.gz"
    return out_dir(subject) / "seg.nii.gz"


@functools.lru_cache(maxsize=16)
def _seg(subject, space):
    image = nib.load(_seg_file(subject, space))
    return np.asarray(image.dataobj).astype(np.int16), image


@functools.lru_cache(maxsize=4)
def _brain(first, space):
    # where a plan can reach: the template brain, or the follow-up scan's own.
    # shared and cached, so callers must not write into it
    if space == "mni":
        return template_brain()
    return np.asarray(nib.load(out_dir(later_of(first)) / "brainmask.nii.gz").dataobj) > 0


def _in(data, region):
    return np.isin(data, REGIONS[region])


@functools.lru_cache(maxsize=16)
def _far(subject, region, space):
    # millimeters from the nearest voxel of the first tumor's region
    data, image = _seg(subject, space)
    seed = _in(data, region)
    return distance(seed if seed.any() else data > 0, image)


def mask(subject, region, space):
    data, _ = _seg(subject, space)
    return _in(data, region)


def pattern(share):
    return next(name for floor, name in PATTERNS if share > floor or floor == 0.0)


def overlap(first, later, region, space):
    # the same compartment at both timepoints, and how far the later one reaches
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
    # no dose is available, so the two NRG volumes stand in for it: 2 cm from the
    # whole tumor is the 46 Gy one, 2 cm from the core the 60 Gy one, and the
    # literature's pattern classes are defined against the 60 Gy volume
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
                # the margin grown from the whole first tumor that would have
                # covered 95% of this, and all of it
                "margin_95_mm": round(float(np.percentile(whole, 95)), 1),
                "margin_max_mm": round(float(whole.max()), 1)})
    return out


def reach(first, later, region, space):
    # how far each voxel of the later tumor sits from the whole first tumor,
    # sorted. Outside the brain counts as never reached, since a plan stops
    # at the brain edge. None when that part is absent at the later scan
    b, _ = _seg(later, space)
    now = _in(b, region)
    if not now.any():
        return None
    return np.sort(np.where(_brain(first, space)[now], _far(first, "WT", space)[now], np.inf))


@functools.lru_cache(maxsize=1)
def _participants():
    # the dataset's own record for each scan. Its interval beats one rebuilt from
    # TCIA study dates, which is wrong for 00130: that patient has an extra study
    # 11 days before the one the release curated as its follow-up
    path = OUTPUTS / "participants.tsv"
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
    # the second resection's share of tissue with malignant features,
    # 1 = under 10% up to 6 = over 90%
    return _followup(first, "PsP_TP_score")


def voxel_ml(first, space):
    return ml_per_voxel(_seg(first, space)[1])


def margin_ml(first, mm, space):
    # brain added by growing the whole first tumor mm millimeters, in mL
    _, image = _seg(first, space)
    far = np.sort(_far(first, "WT", space)[_brain(first, space)])
    grown = np.searchsorted(far, mm, side="right") - np.searchsorted(far, 0, side="right")
    return grown * ml_per_voxel(image)


def zones(first, later, region, space):
    # mL of the later tumor inside the whole first tumor, within the margin of it,
    # past the margin but joined to tumor within it, and in separate lesions lying
    # wholly past it. None when that part is absent at the later scan
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
    # every patient with both timepoints segmented and in that space. Anything
    # that builds a target needs the first scan's tissue map too, and asking for
    # one that is missing quietly runs SynthSeg, so those callers pass tissue=True
    have = {d.name for d in OUTPUTS.iterdir() if _seg_file(d.name, space).exists()
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
    # each cached distance map is a whole volume, so drop them between patients
    _seg.cache_clear()
    _far.cache_clear()
    _brain.cache_clear()
