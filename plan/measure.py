# Where the recurrence sits relative to the first tumor, in MNI, the frame both share.

import functools

import nibabel as nib
import numpy as np
from scipy import ndimage

from paths import LABELS, MNI, out_dir
from yale.atlas import label_volume, parcels

# the margin the recurrence is scored against; RTOG-style, grown from the whole
# segmentation including edema, so it is the phase-1 volume not the 60 Gy one
# why: 90-95% of recurrences fall within 2 cm of the GTV margin [recur21]
MARGIN_MM = 20


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
    far = ndimage.distance_transform_edt(data == 0, sampling=image.header.get_zooms()[:3])
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
