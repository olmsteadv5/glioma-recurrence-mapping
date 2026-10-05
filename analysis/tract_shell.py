#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Does tumor come back more often along the white matter tracts that leave it?
# At each distance from the pre-op whole tumor, 1 mm at a time, compare how often
# white matter recurs where a tract through the tumor passes, against white matter
# at the same distance where none does. White against white, since tracts only run
# through white matter and swelling spreads there too. MNI only: the tracts are an
# MNI atlas.

import csv
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from analysis.capture_curves import FAINT, INK, PATIENT, save, style
from analysis.walkdown import Lines
from geometry import distance
from paths import out_dir, results_dir, walk_dir

# why: SynthSeg's cerebral white matter, left and right [billot23]
WHITE = (2, 41)
SHELLS = tuple(range(20))
# a shell side with fewer voxels than this gives no rate worth reporting
SMALLEST = 20
PARTS = (("ET", "Enhancing recurrence"), ("WT", "Whole-tumor recurrence"))
ON, OFF = "#2a78d6", "#8a9199"


def tract_voxels(first, lines, image, shape):
    # every voxel a centerline through the pre-op whole tumor passes, and its
    # immediate neighbours, since a centerline is one voxel wide
    wt = measure.mask(first, "WT", "mni")
    hit = np.zeros(shape, bool)
    inverse = np.linalg.inv(image.affine)
    for line, inside in zip(lines.lines, lines.lookup(wt)):
        if inside.any():
            ijk = np.rint(nib.affines.apply_affine(inverse, line)).astype(int)
            ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
            hit[tuple(ijk[ok].T)] = True
    return ndimage.binary_dilation(hit)


def rows(first, later, lines):
    data, image = measure._seg(first, "mni")
    wt = measure.mask(first, "WT", "mni")
    # the same brain the walk-down may use: inside it, and the falx rule
    allowed = np.asarray(nib.load(walk_dir(first) / "walk_allowed_mni.nii.gz").dataobj) > 0
    white = np.isin(np.asarray(nib.load(out_dir(first) / "tissue_in_mni.nii.gz").dataobj), WHITE)
    tract = tract_voxels(first, lines, image, data.shape)
    far = distance(wt, image)
    out = []
    for mm in SHELLS:
        shell = (far > mm) & (far <= mm + 1) & allowed & white
        on, off = shell & tract, shell & ~tract
        row = {"patient": first[-8:-3], "pathology": measure.pathology(first), "shell_mm": mm + 1,
               "on_voxels": int(on.sum()), "off_voxels": int(off.sum())}
        for code, _ in PARTS:
            now = measure.mask(later, code, "mni")
            for side, mask in (("on", on), ("off", off)):
                row[f"{code}_{side}_pct"] = (round(100 * float(now[mask].mean()), 3)
                                             if now.any() and mask.sum() >= SMALLEST else None)
        out.append(row)
    return out


def write():
    lines, table = None, []
    for first, later in measure.pairs("mni"):
        if lines is None:
            data, image = measure._seg(first, "mni")
            lines = Lines(image, data.shape)
        table += rows(first, later, lines)
        measure.forget()
    path = results_dir("mni") / "tract_shell.csv"
    with open(path, "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=list(table[0]))
        out.writeheader()
        out.writerows(table)
    return path


def figure():
    # top row: the share of white matter that recurs, on tracts and off, by
    # distance. Bottom row: each patient's difference, on minus off
    with open(results_dir("mni") / "tract_shell.csv") as fh:
        table = list(csv.DictReader(fh))
    num = lambda v: None if v in ("", "None") else float(v)
    shells = sorted({int(r["shell_mm"]) for r in table})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharex=True)
    for j, (code, label) in enumerate(PARTS):
        per = {}
        for r in table:
            per.setdefault(r["patient"], {})[int(r["shell_mm"])] = r
        on = np.array([[num(per[p][s][f"{code}_on_pct"]) if num(per[p][s][f"{code}_on_pct"]) is not None
                        else np.nan for s in shells] for p in per])
        off = np.array([[num(per[p][s][f"{code}_off_pct"]) if num(per[p][s][f"{code}_off_pct"]) is not None
                         else np.nan for s in shells] for p in per])
        keep = ~np.all(np.isnan(on), axis=1)
        on, off = on[keep], off[keep]
        ax = axes[0, j]
        for values, color, name in ((off, OFF, "white matter, no tract"), (on, ON, "white matter on a tract")):
            low, mid, high = np.nanpercentile(values, [25, 50, 75], axis=0)
            ax.fill_between(shells, low, high, color=color, alpha=0.15, lw=0)
            ax.plot(shells, mid, color=color, lw=2.2, label=name)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{label}: share of white matter that recurs (%)", loc="left", fontsize=11, color=INK)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax = axes[1, j]
        diff = on - off
        for row in diff:
            ax.plot(shells, row, color=PATIENT, lw=0.6, alpha=0.6)
        low, mid, high = np.nanpercentile(diff, [25, 50, 75], axis=0)
        ax.fill_between(shells, low, high, color=ON, alpha=0.15, lw=0)
        ax.plot(shells, mid, color=ON, lw=2.4)
        ax.axhline(0, color=FAINT, lw=1, ls=(0, (4, 3)))
        higher = [int(np.sum(diff[:, k] > 0)) for k in range(len(shells))]
        n = [int(np.sum(~np.isnan(diff[:, k]))) for k in range(len(shells))]
        ax.set_title(f"On a tract minus off, per patient (percentage points)\n"
                     f"higher on the tract in {higher[0]} of {n[0]} at 1 mm, "
                     f"{higher[4]} of {n[4]} at 5 mm, {higher[14]} of {n[14]} at 15 mm",
                     loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("Distance from the pre-op whole tumor (mm, 1 mm shells)")
        span = np.nanpercentile(np.abs(diff), 95)
        ax.set_ylim(-span, span)
    axes[0, 0].set_xlim(shells[0], shells[-1])
    fig.text(0.01, 0.005, "Each shell is the white matter between d-1 and d mm from the pre-op whole tumor, "
             "inside the brain and on the tumor's side unless it already crosses (NRG falx rule).\n"
             "On a tract = within one voxel of an atlas centerline through the tumor. Line = median patient, "
             "band = middle 50 percentile, grey = one patient. MNI space.", color=FAINT, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.12, hspace=0.42, wspace=0.18)
    path = results_dir("mni", "figures") / "tract_shell.png"
    save(fig, path)
    return path


def main():
    style()
    print(write(), figure())


if __name__ == "__main__":
    main()
