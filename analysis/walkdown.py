#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# A target that follows white matter instead of growing evenly: walk d mm along
# every atlas tract that leaves the tumor, then wrap one convex hull around the
# tumor and where the walks ended. Built for every d from 0 to 40 mm, for both
# NRG phases, in MNI where the tracts are, and pulled into each follow-up scan
# for patient space.

import csv
import os
import subprocess
import sys
from pathlib import Path

# set before numpy loads, or each worker starts its own pool of threads
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import nibabel as nib
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from geometry import MARGIN_MM, distance, ml_per_voxel, outline, template_brain
from paths import THREADS, anatomy, modality, out_dir, pair_dir, results_dir, stripped_dir, walk_dir
from show.compare import _inverse
from hcp import streamlines

# why: the NRG margin, 2 cm along the tract instead of in every direction
WALK_MM = MARGIN_MM
# twice that, so the walk-down can be followed past where it catches as much as
# the 2 cm volume does, and the brain it saves read off at equal coverage
LONGEST_MM = 2 * WALK_MM
# every whole mm up to it, so nothing between two samples goes unseen
DISTANCES = tuple(range(LONGEST_MM + 1))
# a voxel no walk ever reaches
NEVER = 255
# a tract that leaves and comes back within this is running along the edge,
# not leaving. A judgement call, about three voxels; not from any paper
SKIM_MM = 3
# why: the falx is a barrier unless the tumor already crosses; less than
# this across the midline is taken as registration slop, not crossing. Ours
CROSSED_ML = 0.1

# voxel-face pairs tested at once when filling a hull
BATCH = 20_000_000

# the two NRG phases: the 46 Gy volume grows from the whole abnormality, the
# 60 Gy boost from enhancement and cavity, which on a pre-op scan is the core
PHASES = (("46", "WT"), ("60", "TC"))


class Lines:
    # every streamline point's voxel on the MNI grid, found once: every patient's
    # MNI masks share that grid, so a mask lookup is then a single index
    def __init__(self, image, shape):
        self.lines = streamlines.streamlines()
        stacked = np.vstack(self.lines)
        self.cut = np.cumsum([len(l) for l in self.lines])[:-1]
        ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(image.affine), stacked)).astype(int)
        self.ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
        self.ijk = tuple(ijk[self.ok].T)

    def lookup(self, mask):
        flags = np.zeros(len(self.ok), bool)
        flags[self.ok] = mask[self.ijk]
        return np.split(flags, self.cut)


def _walk(line, inside, allowed, skim=SKIM_MM):
    # one path for every place this line leaves the tumor, in both directions
    # along it, as its points and the distance walked to each. A path stops at
    # the longest walk, at the line's end, at the edge of where a plan may go, or where it
    # comes back into the tumor
    paths = []
    for points, gtv, ok in ((line, inside, allowed), (line[::-1], inside[::-1], allowed[::-1])):
        step = np.linalg.norm(np.diff(points, axis=0), axis=1)
        for i in np.flatnonzero(gtv[:-1] & ~gtv[1:]):
            back = np.flatnonzero(gtv[i + 1:])
            if back.size and step[i:i + 1 + back[0]].sum() < skim:
                continue
            j, walked = i, [0.0]
            while j + 1 < len(points) and ok[j + 1] and not gtv[j + 1] and walked[-1] < LONGEST_MM:
                walked.append(walked[-1] + step[j])
                j += 1
            walked = np.array(walked)
            paths.append((points[i:j + 1], walked))
    return paths


def _ends(paths, mm):
    # where each path stands after mm of walking, or where it stopped if sooner
    out = []
    for points, walked in paths:
        at = min(mm, walked[-1])
        out.append([np.interp(at, walked, points[:, k]) for k in range(3)])
    return np.array(out).reshape(-1, 3)


def _hull(points, image, shape, candidates=None):
    # every voxel whose center is inside the convex hull of the points. With
    # candidates, a boolean volume, only those voxels are tested, which is all a
    # caller growing a target needs: the rest are settled already
    hull = ConvexHull(points)
    inverse = np.linalg.inv(image.affine)
    corners = nib.affines.apply_affine(inverse, points[hull.vertices])
    low = np.maximum(np.floor(corners.min(0)).astype(int), 0)
    high = np.minimum(np.ceil(corners.max(0)).astype(int) + 1, shape)
    if candidates is None:
        box = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(low, high)], indexing="ij"), -1)
        box = box.reshape(-1, 3)
    else:
        window = tuple(slice(a, b) for a, b in zip(low, high))
        box = np.argwhere(candidates[window]) + low
    out = np.zeros(shape, bool)
    # each voxel is tested against every face at once, so the batch shrinks as
    # the hull gains faces, keeping that test near 160 MB
    batch = max(BATCH // len(hull.equations), 1)
    for start in range(0, len(box), batch):
        chunk = box[start:start + batch]
        world = nib.affines.apply_affine(image.affine, chunk)
        keep = np.all(world @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-6, axis=1)
        out[tuple(chunk[keep].T)] = True
    return out


def allowed_mni(first):
    # Where a plan may go, and from what distance. The tumor's own hemisphere is
    # open; across the midline a voxel may join a target grown d mm only if it is
    # within d of the part of that phase's tumor already across, so nothing grows
    # over from the home side. The gate holds that distance, 0 at home and NEVER
    # where nothing may go, for each phase; every target reads the same gate.
    data, image = measure._seg(first, "mni")
    brain = template_brain()
    x = nib.affines.apply_affine(image.affine, np.indices(data.shape).reshape(3, -1).T)[:, 0].reshape(data.shape)
    tumor = data > 0
    # the midline plane itself is home on either side; across means strictly past it
    mostly_left = (tumor & (x < 0)).sum() >= (tumor & (x > 0)).sum()
    home = x <= 0 if mostly_left else x >= 0
    ml = ml_per_voxel(image)
    gates, crossed = {}, False
    for phase, region in PHASES:
        gtv = measure.mask(first, region, "mni")
        if not gtv.any():
            gtv = tumor
        across = gtv & ~home
        gate = np.where(home, 0, NEVER).astype(np.uint8)
        if across.sum() * ml >= CROSSED_ML:
            far = np.ceil(distance(across, image))
            gate[~home & (far <= LONGEST_MM)] = far[~home & (far <= LONGEST_MM)]
            crossed = crossed or phase == "46"
        gates[phase] = gate
    return brain & (gates["46"] != NEVER), gates, crossed


def reach_mni(first, lines, skim=SKIM_MM):
    # for both phases, the shortest walk at which each voxel joins the target.
    # The target at d is every voxel reached by a walk of d or less, so it can
    # only grow with d; the tumor itself is reached at 0
    data, image = measure._seg(first, "mni")
    allowed, gates, crossed = allowed_mni(first)
    may = lines.lookup(allowed)
    reach, info = {}, {"crossed": crossed, "gates": gates}
    for phase, region in PHASES:
        gtv = measure.mask(first, region, "mni")
        if not gtv.any():
            gtv = data > 0
        paths, used = [], 0
        for line, inside, ok in zip(lines.lines, lines.lookup(gtv), may):
            if inside.any():
                used += 1
                paths += _walk(line, inside, ok, skim)
        edge = nib.affines.apply_affine(image.affine, np.argwhere(outline(gtv)))
        first_in = np.full(gtv.shape, NEVER, np.uint8)
        first_in[gtv] = 0
        for mm in DISTANCES:
            # only voxels the plan may reach at this distance and no shorter
            # walk has reached
            open_ = allowed & (gates[phase] <= mm) & (first_in == NEVER)
            first_in[_hull(np.vstack([edge, _ends(paths, mm)]), image, gtv.shape, open_)] = mm
        if phase == "60":
            # the boost stays inside the 46 Gy volume at the same walk
            first_in = np.where(gtv, 0, np.maximum(first_in, reach["46"])).astype(np.uint8)
        reach[phase] = first_in
        info[phase] = {"tracts": used, "exits": len(paths)}
        info[f"ends_{phase}"] = _ends(paths, WALK_MM)
    return reach, allowed, image, info


def build_mni(first, lines):
    # the reach maps, the 20 mm targets the viewer draws, and their end points
    reach, allowed, image, info = reach_mni(first, lines)
    here = walk_dir(first)
    for phase, _ in PHASES:
        nib.save(nib.Nifti1Image(reach[phase], image.affine), here / f"walk_{phase}_reach_mni.nii.gz")
        nib.save(nib.Nifti1Image((reach[phase] <= WALK_MM).astype(np.uint8), image.affine),
                 here / f"walk_{phase}_mni.nii.gz")
        np.save(here / f"walk_{phase}_ends.npy", info.pop(f"ends_{phase}"))
        nib.save(nib.Nifti1Image(info["gates"][phase], image.affine), here / f"walk_gate_{phase}_mni.nii.gz")
    info.pop("gates")
    nib.save(nib.Nifti1Image(allowed.astype(np.uint8), image.affine), here / "walk_allowed_mni.nii.gz")
    return info


def to_native(first, later, name):
    # MNI -> the pre-op scan -> the follow-up scan, the same road the pre-op tumor
    # takes into patient space, so the two line up
    source = walk_dir(first) / f"{name}_mni.nii.gz"
    dest = walk_dir(later) / f"{name}.nii.gz"
    subprocess.run(["greedy", "-d", "3", "-rf", str(anatomy(stripped_dir(later), later)),
                    "-ri", "LABEL", "0.2vox", "-rm", str(source), str(dest),
                    "-r", str(pair_dir(later) / "rigid_first_to_later.mat"),
                    str(out_dir(first) / "affine.mat") + ",-1", str(_inverse(first)),
                    "-threads", THREADS], capture_output=True, check=True)
    return np.asarray(nib.load(dest).dataobj)


def score(first, later, space, reach=None, allowed=None):
    # every target at every walk distance, in one space: the walk-down, the plain
    # margin grown the same distance, and a ball of exactly the walk-down's size.
    # reach and allowed, when given, are MNI maps already in memory. All three
    # obey the same midline gate
    if space == "mni":
        load = lambda name: np.asarray(nib.load(walk_dir(first) / f"{name}_mni.nii.gz").dataobj)
    else:
        load = lambda name: to_native(first, later, name)
    if reach is None:
        reach = {phase: load(f"walk_{phase}_reach") for phase, _ in PHASES}
        allowed = load("walk_allowed") > 0
    gates = {phase: load(f"walk_gate_{phase}") for phase, _ in PHASES}
    allowed = allowed & measure._brain(first, space)
    _, image = measure._seg(first, space)
    ml = measure.voxel_ml(first, space)
    now = {code: measure.mask(later, code, space) for code in ("ET", "WT")}
    rows, iso_46 = [], None
    for phase, region in PHASES:
        gtv = measure.mask(first, region, space)
        if not gtv.any():
            gtv = measure.mask(first, "WT", space)
        first_in = np.where(gtv, 0, np.where(allowed, reach[phase], NEVER))
        # across the midline a voxel is only as near as the tumor already there
        far = np.where(allowed | gtv, np.maximum(distance(gtv, image), gates[phase]), np.inf)
        order = np.argsort(far, axis=None, kind="stable")
        # cumulative catches along the ball's order, so any size is one lookup
        along = {code: np.cumsum(now[code].ravel()[order]) for code in now}
        for mm in DISTANCES:
            walk = first_in <= mm
            iso = far <= mm
            if phase == "60":
                iso &= iso_46[mm]
            if mm == WALK_MM:
                # the NRG volume these tables score, kept so the viewer draws this
                # one; in patient space the walk-down too, exactly as scored
                saved = {"nrg": iso} if space == "mni" else {"nrg": iso, "walk": walk}
                for kind, mask in saved.items():
                    name = f"{kind}_{phase}_mni.nii.gz" if space == "mni" else f"{kind}_{phase}.nii.gz"
                    nib.save(nib.Nifti1Image(mask.astype(np.uint8), image.affine), walk_dir(later) / name)
            size = int(walk.sum())
            for kind, target in (("walk", walk), ("iso", iso), ("ball", None)):
                count = size if target is None else int(target.sum())
                inside_gtv = int(gtv.sum())
                row = {"patient": first[-8:-3], "pathology": measure.pathology(first), "phase": phase,
                       "walk_mm": mm, "target": kind, "volume_ml": round(count * ml, 1),
                       "beyond_gtv_ml": round(max(count - inside_gtv, 0) * ml, 1)}
                for code, mask in now.items():
                    if not mask.any():
                        row[f"{code}_pct"] = None
                        continue
                    caught = (int(along[code][size - 1]) if target is None
                              else int((mask & target).sum()))
                    row[f"{code}_pct"] = round(100 * caught / float(mask.sum()), 1)
                rows.append(row)
        if phase == "46":
            iso_46 = {mm: far <= mm for mm in DISTANCES}
    return rows


def write(space):
    # every distance into walkdown_curves.csv, and 20 mm alone into walkdown.csv
    rows = []
    for first, later in measure.pairs(space):
        rows += score(first, later, space)
        measure.forget()
    for name, keep in (("walkdown_curves.csv", rows),
                       ("walkdown.csv", [r for r in rows if r["walk_mm"] == WALK_MM])):
        with open(results_dir(space) / name, "w", newline="") as fh:
            out = csv.DictWriter(fh, fieldnames=list(keep[0]))
            out.writeheader()
            out.writerows(keep)
    return results_dir(space) / "walkdown_curves.csv"


def _rows(space):
    with open(results_dir(space) / "walkdown.csv") as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        by.setdefault((r["phase"], r["target"]), {})[r["patient"]] = r
    return by


def _num(value):
    return None if value in ("", "None") else float(value)


def summarize(space):
    # medians per target, and the walk against the other two, patient by patient
    from scipy.stats import wilcoxon
    by = _rows(space)
    path = results_dir(space) / "walkdown_summary.csv"
    with open(path, "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["phase", "target", "n", "median_volume_ml", "median_beyond_gtv_ml",
                      "median_ET_pct", "median_WT_pct", "vs_walk", "ET_walk_better",
                      "ET_walk_worse", "ET_p", "WT_walk_better", "WT_walk_worse", "WT_p"])
        for phase, _ in PHASES:
            walk = by[(phase, "walk")]
            for kind in ("walk", "iso", "ball"):
                group = by[(phase, kind)]
                row = [phase, kind, len(group)]
                for key in ("volume_ml", "beyond_gtv_ml", "ET_pct", "WT_pct"):
                    values = [_num(r[key]) for r in group.values() if _num(r[key]) is not None]
                    row.append(round(float(np.median(values)), 1))
                row.append("" if kind == "walk" else f"walk - {kind}")
                for code in ("ET", "WT"):
                    if kind == "walk":
                        row += ["", "", ""]
                        continue
                    pairs = [(_num(walk[p][f"{code}_pct"]), _num(group[p][f"{code}_pct"]))
                             for p in walk if _num(walk[p][f"{code}_pct"]) is not None]
                    diff = np.array([a - b for a, b in pairs])
                    p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                    row += [int((diff > 0.05).sum()), int((diff < -0.05).sum()), f"{p:.3g}"]
                out.writerow(row)
    return path


def summarize_curves(space):
    # at every distance: the median of each target, and how far the walk-down
    # beats a ball of its own size, with a bootstrap 95% interval on that median
    with open(results_dir(space) / "walkdown_curves.csv") as fh:
        rows = list(csv.DictReader(fh))
    cell = {}
    for r in rows:
        cell.setdefault((r["phase"], int(r["walk_mm"]), r["target"]), {})[r["patient"]] = r
    rng = np.random.default_rng(0)
    path = results_dir(space) / "walkdown_curves_summary.csv"
    with open(path, "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["phase", "walk_mm", "target", "median_beyond_gtv_ml", "median_ET_pct",
                      "median_WT_pct", "ET_walk_minus_ball", "ET_low", "ET_high",
                      "WT_walk_minus_ball", "WT_low", "WT_high"])
        for phase, _ in PHASES:
            for mm in DISTANCES:
                walk = cell[(phase, mm, "walk")]
                for kind in ("walk", "iso", "ball"):
                    group = cell[(phase, mm, kind)]
                    row = [phase, mm, kind]
                    for key in ("beyond_gtv_ml", "ET_pct", "WT_pct"):
                        values = [_num(r[key]) for r in group.values() if _num(r[key]) is not None]
                        row.append(round(float(np.median(values)), 1))
                    for code in ("ET", "WT"):
                        if kind != "ball":
                            row += ["", "", ""]
                            continue
                        diff = np.array([_num(walk[p][f"{code}_pct"]) - _num(group[p][f"{code}_pct"])
                                         for p in walk if _num(walk[p][f"{code}_pct"]) is not None])
                        boot = np.median(rng.choice(diff, (2000, len(diff))), axis=1)
                        row += [round(float(np.median(diff)), 2),
                                round(float(np.percentile(boot, 2.5)), 2),
                                round(float(np.percentile(boot, 97.5)), 2)]
                    out.writerow(row)
    return path


def matched(space):
    # the walk-down at equal coverage: for each patient, the shortest walk that
    # catches at least as much as the NRG 2 cm volume, and the brain that saves.
    # No weighting of misses against brain is needed, since none are traded
    with open(results_dir(space) / "walkdown_curves.csv") as fh:
        rows = list(csv.DictReader(fh))
    cell = {(r["patient"], r["phase"], int(r["walk_mm"]), r["target"]): r for r in rows}
    out = []
    for patient in sorted({r["patient"] for r in rows}):
        for phase, _ in PHASES:
            nrg = cell[(patient, phase, WALK_MM, "iso")]
            for code in ("ET", "WT"):
                goal = _num(nrg[f"{code}_pct"])
                if goal is None:
                    continue
                hit = next((mm for mm in DISTANCES
                            if _num(cell[(patient, phase, mm, "walk")][f"{code}_pct"]) >= goal), None)
                walk = cell[(patient, phase, hit, "walk")] if hit is not None else None
                out.append({"patient": patient, "pathology": nrg["pathology"], "phase": phase,
                            "recurrence": code, "nrg_pct": goal,
                            "nrg_beyond_gtv_ml": _num(nrg["beyond_gtv_ml"]),
                            "walk_mm_needed": hit,
                            "walk_beyond_gtv_ml": _num(walk["beyond_gtv_ml"]) if walk else None,
                            "saved_ml": (round(_num(nrg["beyond_gtv_ml"]) - _num(walk["beyond_gtv_ml"]), 1)
                                         if walk else None)})
    path = results_dir(space) / "walkdown_matched.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    return path


def _build(first):
    # one patient's MNI maps; a process of its own so patients run side by side
    data, image = measure._seg(first, "mni")
    return first, build_mni(first, Lines(image, data.shape))


def main():
    from concurrent.futures import ProcessPoolExecutor
    firsts = [first for first, _ in measure.pairs("mni")]
    # one thread each, so the workers share the cores instead of fighting over them
    with ProcessPoolExecutor(8) as pool:
        for first, info in pool.map(_build, firsts):
            print(first[-8:-3], info, flush=True)
    for space in measure.SPACES:
        print(write(space), summarize(space), summarize_curves(space), matched(space), flush=True)


if __name__ == "__main__":
    main()
