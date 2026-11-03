#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: each patient's own midline on the pre-op grid, from where SynthSeg's left and right labels meet, checked against a mirror plane

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.interpolate import RBFInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan.register import _greedy
from paths import PAIRS, THREADS, modality, out_dir, stripped_dir, subject_id

LEFT = (2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28)
RIGHT = (41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60)
# decision: SynthSeg labels near the tumor are guesses, so the surface skips them and fills in from around
TUMOR_GAP_MM = 10
# decision: only trust a crossing that sits close to both hemispheres, not out past one edge of the brain
NEAR_MM = 6
# decision: past this the two midline estimates disagree more than registration slop
DISAGREE_MM = 5


def work(subject):
    path = out_dir(subject) / "midline"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _lateral_axis(image):
    # explainer: the voxel axis that runs left to right, and the sign that turns voxel steps into RAS x
    axis = int(np.argmax(np.abs(image.affine[0, :3])))
    return axis, float(np.sign(image.affine[0, axis]))


def _world_x(image, shape):
    return nib.affines.apply_affine(image.affine, np.indices(shape).reshape(3, -1).T)[:, 0].reshape(shape)


def boundary(subject):
    # explainer: x of the left/right boundary for every column across the head, in RAS mm, nan where no column of brain
    image = nib.load(out_dir(subject) / "tissue.nii.gz")
    tissue = np.asarray(image.dataobj)
    tumor = np.asarray(nib.load(out_dir(subject) / "seg.nii.gz").dataobj) > 0
    zooms = image.header.get_zooms()[:3]
    to_left = ndimage.distance_transform_edt(~np.isin(tissue, LEFT), sampling=zooms)
    to_right = ndimage.distance_transform_edt(~np.isin(tissue, RIGHT), sampling=zooms)
    near_tumor = ndimage.distance_transform_edt(~tumor, sampling=zooms) <= TUMOR_GAP_MM
    axis, sign = _lateral_axis(image)
    x = _world_x(image, tissue.shape)
    # explainer: put the lateral axis first so every column is one row of a 2d sheet
    f = np.moveaxis(to_left - to_right, axis, 0)
    gap = np.moveaxis(np.minimum(to_left, to_right), axis, 0)
    skip = np.moveaxis(near_tumor, axis, 0)
    xs = np.moveaxis(x, axis, 0)
    brain = np.moveaxis(tissue > 0, axis, 0)
    crossing = np.sign(f[:-1]) != np.sign(f[1:])
    crossing &= (gap[:-1] <= NEAR_MM) & brain[:-1]
    seen = np.full(f.shape[1:], np.nan)
    kept = np.full(f.shape[1:], np.nan)
    columns = np.argwhere(crossing.any(0))
    centre = np.median(xs[:-1][crossing])
    for j, k in columns:
        hits = np.flatnonzero(crossing[:, j, k])
        # explainer: linear interpolation to where f is zero between the two voxels
        at = [xs[i, j, k] + (xs[i + 1, j, k] - xs[i, j, k]) * f[i, j, k] / (f[i, j, k] - f[i + 1, j, k])
              for i in hits]
        # decision: a column with several crossings keeps the one nearest the head's median midline
        best = int(np.argmin(np.abs(np.array(at) - centre)))
        seen[j, k] = at[best]
        if not skip[hits[best], j, k]:
            kept[j, k] = at[best]
    head = brain.any(0)
    return seen, kept, head, image, tissue, tumor


def surface(kept, head, smooth=50.0, step=4):
    # explainer: thin plate fit through the trusted boundary points, smooth enough to iron out label noise, extrapolates over the tumor gap
    have = np.argwhere(~np.isnan(kept))
    have = have[(have[:, 0] % step == 0) & (have[:, 1] % step == 0)]
    fit = RBFInterpolator(have.astype(float), kept[tuple(have.T)], kernel="thin_plate_spline", smoothing=smooth)
    every = np.argwhere(np.ones(kept.shape, bool))
    out = fit(every.astype(float)).reshape(kept.shape)
    return out


def lateral(subject, sheet, image, shape):
    # explainer: signed mm from the midline along RAS x, negative is the left hemisphere
    axis, _ = _lateral_axis(image)
    x = _world_x(image, shape)
    return x - np.expand_dims(sheet, axis)


def mirror(subject, tumor, image):
    # explainer: flip the T2 and fit it back onto itself with the tumor and its mirror image masked, the fit's midpoints lie on the symmetry plane
    w = work(subject)
    t2 = nib.load(modality(stripped_dir(subject), subject, "T2"))
    axis, _ = _lateral_axis(t2)
    data = np.asarray(t2.dataobj).astype(np.float32)
    nib.save(nib.Nifti1Image(np.flip(data, axis), t2.affine), w / "t2_flipped.nii.gz")
    gone = ndimage.binary_dilation(tumor | np.flip(tumor, axis), iterations=TUMOR_GAP_MM)
    nib.save(nib.Nifti1Image(((data > 0) & ~gone).astype(np.uint8), t2.affine), w / "mask.nii.gz")
    log = w / "mirror.log"
    pair = ["-i", modality(stripped_dir(subject), subject, "T2"), w / "t2_flipped.nii.gz"]
    # explainer: greedy's moments mode only writes the starting guess, the rigid fit runs from it
    _greedy(["-d", 3, "-moments", 1, *pair, "-o", w / "start.mat"], log)
    _greedy(["-d", 3, "-a", "-dof", 6, "-m", "SSD", *pair, "-gm", w / "mask.nii.gz", "-ia", w / "start.mat",
             "-o", w / "mirror.mat", "-n", "100x50x20", "-threads", THREADS], log)
    fit = np.loadtxt(w / "mirror.mat")
    flip = np.diag([1.0, 1.0, 1.0, 1.0])
    flip[axis, axis] = -1
    flip[axis, 3] = data.shape[axis] - 1
    # explainer: the flip in world mm is the voxel flip seen through the header, and fixed point p matches flipped(fit p) = original(flip fit p)
    reflect = t2.affine @ flip @ np.linalg.inv(t2.affine) @ fit
    points = nib.affines.apply_affine(t2.affine, np.argwhere(data > 0)[::50])
    middle = (points + nib.affines.apply_affine(reflect, points)) / 2
    # decision: plane as x = a + b y + c z, the midline is never close to edge-on in x
    design = np.c_[np.ones(len(middle)), middle[:, 1], middle[:, 2]]
    coef, *_ = np.linalg.lstsq(design, middle[:, 0], rcond=None)
    angle = np.degrees(np.arccos(np.clip((np.trace(fit[:3, :3]) - 1) / 2, -1, 1)))
    return coef, angle


def run(number):
    subject = subject_id(number)
    w = work(subject)
    seen, kept, head, image, tissue, tumor = boundary(subject)
    sheet = surface(kept, head)
    side = lateral(subject, sheet, image, tissue.shape).astype(np.float32)
    nib.save(nib.Nifti1Image(side, image.affine), w / "lateral.nii.gz")
    coef, angle = mirror(subject, tumor, image)
    axis, _ = _lateral_axis(image)
    yz = np.moveaxis(nib.affines.apply_affine(image.affine, np.indices(tissue.shape).reshape(3, -1).T)
                     .reshape(*tissue.shape, 3), axis, 0)[0]
    plane = coef[0] + coef[1] * yz[..., 1] + coef[2] * yz[..., 2]
    gap = np.abs(sheet - plane)[head]
    trusted = ~np.isnan(kept)
    ml = float(np.prod(image.header.get_zooms()[:3])) / 1000
    home_left = (tumor & (side < 0)).sum() >= (tumor & (side > 0)).sum()
    across = tumor & ((side > 0) if home_left else (side < 0))
    labelled_across = tumor & np.isin(tissue, RIGHT if home_left else LEFT)
    result = {"tumor_side": "left" if home_left else "right",
              "columns_trusted": round(float(trusted.sum() / max(int((~np.isnan(seen)).sum()), 1)), 3),
              "label_noise_mm": round(float(np.nanmedian(np.abs(kept - sheet))), 2),
              "mirror_rotation_deg": round(float(angle), 2),
              "vs_mirror_median_mm": round(float(np.median(gap)), 2),
              "vs_mirror_p95_mm": round(float(np.percentile(gap, 95)), 2),
              "tumor_across_ml": round(float(across.sum()) * ml, 2),
              "tumor_labelled_other_side_ml": round(float(labelled_across.sum()) * ml, 2)}
    result["flag_disagree"] = result["vs_mirror_median_mm"] > DISAGREE_MM
    result["flag_crosses"] = result["tumor_across_ml"] >= 0.1
    (w / "check.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(w / "sheet.npz", sheet=sheet, seen=seen, kept=kept, plane=plane)
    return result


if __name__ == "__main__":
    numbers = [int(a) for a in sys.argv[1:]] or sorted(int(p.name) for p in PAIRS.iterdir()
                                                       if p.name.isdigit() and int(p.name) < 10000)
    for n in numbers:
        print(n, run(n), flush=True)
