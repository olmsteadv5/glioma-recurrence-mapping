# Where the recurrence sits relative to the first tumor, in MNI, the frame both share.

import datetime
import functools
import json

import nibabel as nib
import numpy as np
from scipy import ndimage

from paths import LABELS, MNI, OUTPUTS, REGIONS, out_dir
from yale.atlas import label_volume, parcels

# the margin the recurrence is scored against; RTOG-style, grown from the whole
# segmentation including edema, so it is the phase-1 volume not the 60 Gy one
# why: 90-95% of recurrences fall within 2 cm of the GTV margin [recur21]
MARGIN_MM = 20

# why: central >95%, in-field 80-95%, marginal 20-80%, distant <20% of the
#      recurrence inside the high-dose volume [lee99]
PATTERNS = ((0.95, "central"), (0.80, "in-field"), (0.20, "marginal"), (0.0, "distant"))


@functools.lru_cache(maxsize=16)
def _seg(subject):
    image = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")
    return np.asarray(image.dataobj).astype(np.int16), image


def _ml(image):
    return float(np.prod(image.header.get_zooms()[:3])) / 1000.0


@functools.lru_cache(maxsize=8)
def grown(subject, mm=MARGIN_MM):
    # the isotropic expansion of the first tumor, clipped to the template brain
    data, image = _seg(subject)
    far = ndimage.distance_transform_edt(data == 0,
                                         sampling=image.header.get_zooms()[:3])
    return (far <= mm) & (np.asarray(nib.load(MNI).dataobj) > 0)


def in_original(first, later):
    # how much of the recurrence lands on tissue the first tumor already held
    a, _ = _seg(first)
    b, image = _seg(later)
    hit, ml = b > 0, _ml(image)
    inside = hit & (a > 0)
    return {"recur_ml": round(float(hit.sum()) * ml, 2),
            "inside_ml": round(float(inside.sum()) * ml, 2),
            "share": round(float(inside.sum()) / max(float(hit.sum()), 1), 4)}


def in_margin(first, later, mm=MARGIN_MM):
    # the number standard of care is judged on: does the expansion cover it
    b, image = _seg(later)
    hit, ml = b > 0, _ml(image)
    ring = grown(first, mm)
    inside = hit & ring
    return {"margin_mm": mm,
            "margin_ml": round(float(ring.sum()) * ml, 2),
            "covered_ml": round(float(inside.sum()) * ml, 2),
            "missed_ml": round(float((hit & ~ring).sum()) * ml, 2),
            "share": round(float(inside.sum()) / max(float(hit.sum()), 1), 4)}


def by_class(first, later):
    # which compartment of the first tumor the recurrence overlaps, if any
    a, _ = _seg(first)
    b, image = _seg(later)
    hit, ml = b > 0, _ml(image)
    total = max(float(hit.sum()), 1)
    out = {name: round(float((hit & (a == value)).sum()) / total, 4)
           for value, name in LABELS.items() if (a == value).any()}
    out["outside"] = round(float((hit & (a == 0)).sum()) / total, 4)
    out["_ml"] = {name: round(float((hit & (a == value)).sum()) * ml, 2)
                  for value, name in LABELS.items() if (a == value).any()}
    return out


def lobes(subject, floor=0.01):
    # the lobes the tumor sits in, by share of its own volume
    data, image = _seg(subject)
    labels, names = label_volume(nib.load(MNI))
    lobe = {p["name"]: p["lobe"] for p in parcels()}
    hit, ml = data > 0, _ml(image)

    counts = np.bincount(labels[hit].ravel(), minlength=len(names) + 1)
    tally = {}
    for index, name in enumerate(names, start=1):
        if counts[index]:
            tally[lobe[name]] = tally.get(lobe[name], 0) + int(counts[index])
    named = max(float(sum(tally.values())), 1)
    return [{"lobe": k, "ml": round(v * ml, 2), "share": round(v / named, 4)}
            for k, v in sorted(tally.items(), key=lambda kv: -kv[1])
            if v / named >= floor]


def table(pairs):
    # one row per pair, the four measures the recurrence question asks for
    rows = []
    for first, later in pairs:
        margin = in_margin(first, later)
        rows.append({"pair": f"{first[-5:]} -> {later[-5:]}",
                     "recur_ml": margin["covered_ml"] + margin["missed_ml"],
                     "in_original": in_original(first, later)["share"],
                     "in_margin": margin["share"],
                     "missed_ml": margin["missed_ml"],
                     "by_class": by_class(first, later),
                     "lobes_first": [l["lobe"] for l in lobes(first)],
                     "lobes_later": [l["lobe"] for l in lobes(later)]})
    return rows


def _in(data, region):
    return np.isin(data, REGIONS[region])


@functools.lru_cache(maxsize=16)
def _far(subject, region):
    # millimeters from the nearest voxel of the first tumor's region
    data, image = _seg(subject)
    seed = _in(data, region)
    return ndimage.distance_transform_edt(~(seed if seed.any() else data > 0),
                                          sampling=image.header.get_zooms()[:3])


def pattern(share):
    return next(name for floor, name in PATTERNS if share > floor or floor == 0.0)


def overlap(first, later, region="WT"):
    # the same compartment at both timepoints, and how far the later one reaches
    a, _ = _seg(first)
    b, image = _seg(later)
    ml = _ml(image)
    was, now = _in(a, region), _in(b, region)
    out = {"region": region, "first_ml": round(float(was.sum()) * ml, 2),
           "later_ml": round(float(now.sum()) * ml, 2)}
    if not now.any():
        return out

    shared = float((was & now).sum())
    brain = np.asarray(nib.load(MNI).dataobj) > 0
    # no dose is available, so the two NRG volumes stand in for it: 2 cm from the
    # whole tumor is the 46 Gy one, 2 cm from the core the 60 Gy one, and the
    # literature's pattern classes are defined against the 60 Gy volume
    whole = _far(first, "WT")[now]
    core = _far(first, "TC")[now]
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


@functools.lru_cache(maxsize=1)
def _dates():
    # study dates from the TCIA query, one list per patient
    path = OUTPUTS / "study_dates.json"
    return json.loads(path.read_text()) if path.exists() else {}


def interval(first):
    dates = _dates().get(first.rsplit("_", 1)[0], [])
    if len(dates) < 2:
        return None
    day = datetime.date.fromisoformat
    return (day(dates[1]) - day(dates[0])).days


def pairs(tissue=False):
    # every patient with both timepoints segmented and in MNI. Anything that
    # builds a target needs the first scan's tissue map too, and asking for one
    # that is missing quietly runs SynthSeg, so those callers pass tissue=True
    have = {d.name for d in OUTPUTS.iterdir() if (d / "seg_in_mni.nii.gz").exists()
            and (not tissue or not d.name.endswith("_11")
                 or (d / "tissue_in_mni.nii.gz").exists())}
    return [(name, name[:-2] + "21") for name in sorted(have)
            if name.endswith("_11") and name[:-2] + "21" in have]


def review(pairs, regions=("WT", "TC", "ET")):
    rows = []
    for first, later in pairs:
        for region in regions:
            rows.append({"patient": first[-8:-3], "days": interval(first),
                         **overlap(first, later, region)})
        _seg.cache_clear()
        _far.cache_clear()
    return rows
