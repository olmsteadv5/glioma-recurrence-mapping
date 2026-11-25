#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: where TumorSynth and robust SynthSeg agree on brain structures, by distance from the tumor edge, and how its tumor compares with ours

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import SCANS, results_dir

# explainer: TumorSynth's 17 structures plus tumor (label 18), no left or right
NAMES = {1: "white matter", 2: "cortex", 3: "lateral ventricle", 4: "inferior lateral ventricle",
         5: "cerebellar white matter", 6: "cerebellar cortex", 7: "thalamus", 8: "caudate", 9: "putamen",
         10: "pallidum", 11: "3rd ventricle", 12: "4th ventricle", 13: "brainstem", 14: "hippocampus",
         15: "amygdala", 16: "accumbens", 17: "ventral diencephalon", 18: "tumor"}
# explainer: SynthSeg label to TumorSynth label, both hemispheres to one, every cortical parcel to cortex, CSF to nothing
SYNTHSEG = {2: 1, 41: 1, 3: 2, 42: 2, 4: 3, 43: 3, 5: 4, 44: 4, 7: 5, 46: 5, 8: 6, 47: 6, 10: 7, 49: 7, 11: 8, 50: 8,
            12: 9, 51: 9, 13: 10, 52: 10, 14: 11, 15: 12, 16: 13, 17: 14, 53: 14, 18: 15, 54: 15, 26: 16, 58: 16,
            28: 17, 60: 17}
# decision: distance bands from the edge of our BraTS whole tumor, where recurrence mostly lands is 0 to 20 mm
BANDS = (("inside tumor", -np.inf, 0), ("0-5 mm", 0, 5), ("5-10 mm", 5, 10), ("10-20 mm", 10, 20),
         ("20-40 mm", 20, 40), ("40+ mm", 40, np.inf))
SHOWN = (1, 2, 3, 7)


def to_tumorsynth(synthseg):
    out = np.zeros(synthseg.shape, np.int16)
    for a, b in SYNTHSEG.items():
        out[synthseg == a] = b
    out[(synthseg >= 1000) & (synthseg < 3000)] = 2
    return out


def dice(a, b):
    total = a.sum() + b.sum()
    return float(2 * (a & b).sum() / total) if total else np.nan


def compare(scan):
    here = SCANS / scan
    image = nib.load(here / "seg.nii.gz")
    tumor = np.asarray(image.dataobj) > 0
    ts = np.asarray(nib.load(here / "tumorsynth.nii.gz").dataobj).astype(np.int16)
    ss = to_tumorsynth(np.asarray(nib.load(here / "tissue_robust.nii.gz").dataobj).astype(np.int32))
    if ts.shape != tumor.shape or ss.shape != tumor.shape:
        raise ValueError(f"{scan}: grids differ {ts.shape} {ss.shape} {tumor.shape}")
    zooms = image.header.get_zooms()[:3]
    # explainer: signed distance, negative inside the tumor, positive outside
    outside = ndimage.distance_transform_edt(~tumor, sampling=zooms)
    inside = ndimage.distance_transform_edt(tumor, sampling=zooms)
    signed = np.where(tumor, -inside, outside)
    brain = (ts > 0) | (ss > 0)
    rows = []
    for name, lo, hi in BANDS:
        band = brain & (signed > lo) & (signed <= hi) if np.isfinite(lo) else brain & tumor
        both = band & (ts > 0) & (ts != 18) & (ss > 0)
        row = {"scan": scan, "band": name, "voxels": int(band.sum()),
               "tumorsynth calls tumor": round(float((ts[band] == 18).mean()), 3) if band.any() else None,
               "same structure": round(float((ts[both] == ss[both]).mean()), 3) if both.any() else None}
        for k in SHOWN:
            row[f"dice {NAMES[k]}"] = round(dice((ts == k) & band, (ss == k) & band), 3)
        rows.append(row)
    summary = {"scan": scan, "tumor dice vs ours": round(dice(ts == 18, tumor), 3),
               "tumorsynth tumor mL": round(float((ts == 18).sum() * np.prod(zooms) / 1000), 1),
               "our tumor mL": round(float(tumor.sum() * np.prod(zooms) / 1000), 1)}
    return rows, summary


def main():
    scans = sorted(p.name for p in SCANS.iterdir() if (p / "tumorsynth.nii.gz").exists()
                   and (p / "tissue_robust.nii.gz").exists())
    rows, summaries = [], []
    for s in scans:
        r, m = compare(s)
        rows += r
        summaries.append(m)
    for name, data in (("tumorsynth_vs_synthseg_by_distance.csv", rows), ("tumorsynth_tumor_vs_ours.csv", summaries)):
        with open(results_dir("native") / name, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(f"{len(scans)} scans")
    for m in summaries:
        print(m)
    for name, *_ in BANDS:
        mine = [r for r in rows if r["band"] == name]
        same = [r["same structure"] for r in mine if r["same structure"] is not None]
        calls = [r["tumorsynth calls tumor"] for r in mine if r["tumorsynth calls tumor"] is not None]
        print(f"{name:13s} same structure {np.mean(same) if same else float('nan'):.3f} | "
              f"tumorsynth says tumor {np.mean(calls) if calls else float('nan'):.3f} | "
              + " | ".join(f"{NAMES[k]} {np.nanmean([r[f'dice {NAMES[k]}'] for r in mine]):.2f}" for k in SHOWN))


if __name__ == "__main__":
    main()
