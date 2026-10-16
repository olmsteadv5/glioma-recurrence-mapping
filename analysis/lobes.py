#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Which lobes each patient's original and recurring tumors sit in, in both spaces.

import csv
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from paths import ATLAS_DIR, MNI, THREADS, anatomy, out_dir, pair_dir, results_dir, stripped_dir
from show.compare import _inverse
from yale import atlas

LOBE_MAP = ATLAS_DIR / "lobes_on_mni.nii.gz"
# a lobe counts toward a tumor once it holds this share of it
FLOOR = 0.05
PARTS = (("WT", "whole"), ("ET", "enhancing"))


def lobe(parcel):
    # the nearest-parcel fill hands the deep gray and ventricles to the callosum
    # parcels, so that class is everything central, not the callosum alone. The
    # cerebellum ends up temporal or occipital, which this cohort barely reaches
    name = "deep / callosal" if parcel["lobe"] == "Corpus callosum" else parcel["lobe"].lower()
    return f"{parcel['hemisphere']} {name}"


def names():
    return sorted({lobe(p) for p in atlas.parcels()})


def mni_lobes():
    # the atlas only labels cortex, so every other voxel takes the lobe of the
    # nearest parcel on its own side of the brain
    if LOBE_MAP.exists():
        return LOBE_MAP
    template = nib.load(MNI)
    labels, _ = atlas.label_volume(template)
    order = names()
    lobe_of = np.zeros(len(atlas.parcels()) + 1, np.int16)
    for i, p in enumerate(atlas.parcels(), 1):
        lobe_of[i] = order.index(lobe(p)) + 1
    lobes = lobe_of[labels]
    x = nib.affines.apply_affine(template.affine, np.indices(labels.shape).reshape(3, -1).T)[:, 0]
    left = (x < 0).reshape(labels.shape)
    out = np.zeros(labels.shape, np.int16)
    for side, here in (("Left", left), ("Right", ~left)):
        mine = np.isin(lobes, [order.index(n) + 1 for n in order if n.startswith(side)])
        _, nearest = ndimage.distance_transform_edt(~mine, return_indices=True)
        filled = lobes[tuple(nearest)]
        out[here] = filled[here]
    nib.save(nib.Nifti1Image(out, template.affine), LOBE_MAP)
    return LOBE_MAP


def native_lobes(later):
    # the MNI lobe map pulled back into the follow-up scan, through that scan's
    # own registration, the same way back show.compare takes for the tumor pieces
    dest = pair_dir(later) / "lobes.nii.gz"
    if not dest.exists():
        subprocess.run(["greedy", "-d", "3", "-rf", str(anatomy(stripped_dir(later), later)),
                        "-ri", "LABEL", "0.2vox", "-rm", str(mni_lobes()), str(dest),
                        "-r", str(out_dir(later) / "affine.mat") + ",-1", str(_inverse(later)),
                        "-threads", THREADS], capture_output=True, check=True)
    return dest


def shares(tumor, lobes):
    counts = np.bincount(lobes[tumor], minlength=len(names()) + 1)[1:]
    total = counts.sum()
    return {} if not total else {n: c / total for n, c in zip(names(), counts) if c / total >= FLOOR}


def spell(found):
    return "; ".join(f"{n} {100 * v:.0f}%" for n, v in sorted(found.items(), key=lambda kv: -kv[1]))


def write(space):
    path = results_dir(space) / "tumor_lobes.csv"
    with open(path, "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["patient", "pathology", "original_whole", "recurrence_whole",
                      "recurrence_enhancing", "lobes_new_in_recurrence"])
        for first, later in measure.pairs(space):
            source = mni_lobes() if space == "mni" else native_lobes(later)
            lobes = np.asarray(nib.load(source).dataobj).astype(np.int16)
            was = shares(measure.mask(first, "WT", space), lobes)
            now = {code: shares(measure.mask(later, code, space), lobes) for code, _ in PARTS}
            new = sorted(set(now["WT"]) - set(was))
            out.writerow([first[-8:-3], measure.pathology(first), spell(was), spell(now["WT"]),
                          spell(now["ET"]) or "none", "; ".join(new) or "none"])
            measure.forget()
    return path


if __name__ == "__main__":
    for space in measure.SPACES:
        print(write(space))
