#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: walk d mm along every tract leaving the tumor, then hull the tumor plus where the walks ended

import csv
import os
import subprocess
import sys
from pathlib import Path

# decision: must be set before numpy loads or every worker starts its own thread pool
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import nibabel as nib
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from geometry import MARGIN_MM, distance, ml_per_voxel, outline, template_brain
from paths import PATIENT_BUILT, REGIONS, THREADS, anatomy, out_dir, pair_dir, results_dir, stripped_dir, walk_dir
from show.compare import _inverse
from hcp import streamlines

# decision: same 2 cm as NRG, measured along the tract instead of in every direction
WALK_MM = MARGIN_MM
# decision: run to double so we can see past where it matches the 2 cm volume
LONGEST_MM = 2 * WALK_MM
DISTANCES = tuple(range(LONGEST_MM + 1))
NEVER = 255
# decision: a tract that leaves and comes back this soon is skimming the edge, not leaving. judgement call
SKIM_MM = 3
# decision: falx is a barrier unless the tumor already crosses, less than this across is registration slop. our number
CROSSED_ML = 0.1

BATCH = 20_000_000

# decision: NRG phases: 46 Gy from the whole abnormality, 60 Gy from enhancement + cavity (the core, pre-op)
PHASES = (("46", "WT"), ("60", "TC"))


class Lines:
    # explainer: each point's voxel is looked up once per grid
    def __init__(self, lines, image, shape):
        self.lines = lines
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
    # explainer: a path stops at the max walk, the line's end, the allowed edge, or back inside the tumor
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
    out = []
    for points, walked in paths:
        at = min(mm, walked[-1])
        out.append([np.interp(at, walked, points[:, k]) for k in range(3)])
    return np.array(out).reshape(-1, 3)


def _hull(points, image, shape, candidates=None):
    # explainer: only candidate voxels get tested, the rest are already settled
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
    # explainer: the batch shrinks as the hull gains faces, keeps memory flat
    batch = max(BATCH // len(hull.equations), 1)
    for start in range(0, len(box), batch):
        chunk = box[start:start + batch]
        world = nib.affines.apply_affine(image.affine, chunk)
        keep = np.all(world @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-6, axis=1)
        out[tuple(chunk[keep].T)] = True
    return out


def world_x(image, shape):
    return nib.affines.apply_affine(image.affine, np.indices(shape).reshape(3, -1).T)[:, 0].reshape(shape)


def grid(first, source):
    # explainer: the atlas walk is built on MNI with x = 0 as midline, the patient walk on the pre-op grid with the patient's own midline
    if source == "atlas":
        data, image = measure._seg(first, "mni")
        return data, image, template_brain(), world_x(image, data.shape)
    here = out_dir(first)
    image = nib.load(here / "seg.nii.gz")
    data = np.asarray(image.dataobj).astype(np.int16)
    brain = np.asarray(nib.load(here / "brainmask.nii.gz").dataobj) > 0
    return data, image, brain, np.asarray(nib.load(here / "midline" / "lateral.nii.gz").dataobj)


def tracts(first, source):
    if source == "atlas":
        return streamlines.streamlines()
    saved = np.load(out_dir(first) / TRACT_RUN[source] / "tracts_native.npz")
    return np.split(saved["points"].astype(float), saved["offsets"][1:-1])


def _gtv(data, region):
    gtv = np.isin(data, REGIONS[region])
    return gtv if gtv.any() else data > 0


def allowed_region(data, image, brain, x):
    # decision: across the midline a voxel joins at d only if it's within d of tumor already across
    tumor = data > 0
    # decision: voxels on the plane count as home, across means strictly past it
    mostly_left = (tumor & (x < 0)).sum() >= (tumor & (x > 0)).sum()
    home = x <= 0 if mostly_left else x >= 0
    ml = ml_per_voxel(image)
    gates, crossed = {}, False
    for phase, region in PHASES:
        gtv = _gtv(data, region)
        across = gtv & ~home
        gate = np.where(home, 0, NEVER).astype(np.uint8)
        if across.sum() * ml >= CROSSED_ML:
            far = np.ceil(distance(across, image))
            gate[~home & (far <= LONGEST_MM)] = far[~home & (far <= LONGEST_MM)]
            crossed = crossed or phase == "46"
        gates[phase] = gate
    return brain & (gates["46"] != NEVER), gates, crossed


def walk_reach(data, image, brain, x, lines, skim=SKIM_MM):
    # explainer: stores the shortest walk at which each voxel joins, so target(d) is reach <= d and only grows
    allowed, gates, crossed = allowed_region(data, image, brain, x)
    may = lines.lookup(allowed)
    reach, info = {}, {"crossed": crossed, "gates": gates}
    for phase, region in PHASES:
        gtv = _gtv(data, region)
        paths, used = [], 0
        for line, inside, ok in zip(lines.lines, lines.lookup(gtv), may):
            if inside.any():
                used += 1
                paths += _walk(line, inside, ok, skim)
        edge = nib.affines.apply_affine(image.affine, np.argwhere(outline(gtv)))
        first_in = np.full(gtv.shape, NEVER, np.uint8)
        first_in[gtv] = 0
        for mm in DISTANCES:
            open_ = allowed & (gates[phase] <= mm) & (first_in == NEVER)
            first_in[_hull(np.vstack([edge, _ends(paths, mm)]), image, gtv.shape, open_)] = mm
        if phase == "60":
            # decision: boost stays inside the 46 Gy volume, NRG cone-down
            first_in = np.where(gtv, 0, np.maximum(first_in, reach["46"])).astype(np.uint8)
        reach[phase] = first_in
        info[phase] = {"tracts": used, "exits": len(paths)}
        info[f"ends_{phase}"] = _ends(paths, WALK_MM)
    return reach, allowed, image, info


# explainer: file tags for the build grid and for the follow-up grid
BUILT = {"atlas": "_mni", "dti": "_preop", "dti_prep": PATIENT_BUILT}
SCORED = {"atlas": "", "dti": "_dti", "dti_prep": "_dtiprep"}
# explainer: which diffusion run each patient source reads its tracts from
TRACT_RUN = {"dti": "diffusion", "dti_prep": "diffusion_prep"}
# decision: rescoring with the fixed ball and the beyond-tumor capture writes beside the tables the lab has seen, never over them
RESCORED = "tables_rescored"
# explainer: set by the covered argument, then targets and recurrence are both cut to where the pre-op diffusion scan reached
COVERED = False
# explainer: set by the predict argument, which cohorts a run scores, each set gets its own tables
COHORTS = ("UPenn",)


def build(first, source="atlas"):
    data, image, brain, x = grid(first, source)
    reach, allowed, image, info = walk_reach(data, image, brain, x, Lines(tracts(first, source), image, data.shape))
    here, tag = walk_dir(first), BUILT[source]
    ends = "" if source == "atlas" else tag
    for phase, _ in PHASES:
        nib.save(nib.Nifti1Image(reach[phase], image.affine), here / f"walk_{phase}_reach{tag}.nii.gz")
        nib.save(nib.Nifti1Image((reach[phase] <= WALK_MM).astype(np.uint8), image.affine),
                 here / f"walk_{phase}{tag}.nii.gz")
        np.save(here / f"walk_{phase}_ends{ends}.npy", info.pop(f"ends_{phase}"))
        nib.save(nib.Nifti1Image(info["gates"][phase], image.affine), here / f"walk_gate_{phase}{tag}.nii.gz")
    info.pop("gates")
    nib.save(nib.Nifti1Image(allowed.astype(np.uint8), image.affine), here / f"walk_allowed{tag}.nii.gz")
    return info


def to_native(first, later, name, source="atlas"):
    # decision: same route the pre-op tumor takes into patient space, so they line up
    moving = walk_dir(first) / f"{name}{BUILT[source]}.nii.gz"
    dest = walk_dir(later) / f"{name}{SCORED[source]}.nii.gz"
    chain = [str(pair_dir(later) / "rigid_first_to_later.mat")]
    if source == "atlas":
        chain += [str(out_dir(first) / "affine.mat") + ",-1", str(_inverse(first))]
    subprocess.run(["greedy", "-d", "3", "-rf", str(anatomy(stripped_dir(later), later)),
                    "-ri", "LABEL", "0.2vox", "-rm", str(moving), str(dest), "-r", *chain,
                    "-threads", THREADS], capture_output=True, check=True)
    return np.asarray(nib.load(dest).dataobj)


def _coverage(first, later):
    # explainer: the pre-op diffusion scan's reach, moved onto the follow-up grid the targets are scored on
    dest = walk_dir(later) / "diffusion_coverage.nii.gz"
    if not dest.exists():
        subprocess.run(["greedy", "-d", "3", "-rf", str(anatomy(stripped_dir(later), later)),
                        "-rm", str(out_dir(first) / "diffusion_prep" / "coverage_on_t2.nii.gz"), str(dest),
                        "-r", str(pair_dir(later) / "rigid_first_to_later.mat"), "-threads", THREADS],
                       capture_output=True, check=True)
    return np.asarray(nib.load(dest).dataobj) >= 0.5


def score(first, later, space, reach=None, allowed=None, source="atlas"):
    # explainer: walk-down, plain margin at the same d, and a ball the walk-down's size, all under the same midline gate
    if space == "mni":
        load = lambda name: np.asarray(nib.load(walk_dir(first) / f"{name}_mni.nii.gz").dataobj)
    else:
        load = lambda name: to_native(first, later, name, source)
    if reach is None:
        reach = {phase: load(f"walk_{phase}_reach") for phase, _ in PHASES}
        allowed = load("walk_allowed") > 0
    gates = {phase: load(f"walk_gate_{phase}") for phase, _ in PHASES}
    allowed = allowed & measure._brain(first, space)
    _, image = measure._seg(first, space)
    ml = measure.voxel_ml(first, space)
    now = {code: measure.mask(later, code, space) for code in ("ET", "WT")}
    if COVERED:
        seen = _coverage(first, later)
        allowed, now = allowed & seen, {code: mask & seen for code, mask in now.items()}
    # decision: recurrence beyond the whole original abnormality is what a margin has to reach, every target covers the rest
    original = measure.mask(first, "WT", space)
    caught = {**now, **{f"{code}_beyond": mask & ~original for code, mask in now.items()}}
    # decision: ties in distance broken at random with a seed per patient, a voxel-order tie-break leans the ball one way
    shuffle = np.random.default_rng(int(first[-8:-3])).random(original.size)
    rows, iso_46, balls_46 = [], None, None
    for phase, region in PHASES:
        gtv = measure.mask(first, region, space)
        if not gtv.any():
            gtv = measure.mask(first, "WT", space)
        first_in = np.where(gtv, 0, np.where(allowed, reach[phase], NEVER))
        far = np.where(allowed | gtv, np.maximum(distance(gtv, image), gates[phase]), np.inf)
        order = np.lexsort((shuffle, far.ravel()))
        sizes = {}
        for mm in DISTANCES:
            walk = first_in <= mm
            iso = far <= mm
            if phase == "60":
                iso &= iso_46[mm]
            if mm == WALK_MM and not COVERED:
                # decision: save the exact NRG volume we scored so the viewer draws that one
                saved = {"nrg": iso} if space == "mni" else {"nrg": iso, "walk": walk}
                for kind, mask in saved.items():
                    name = f"{kind}_{phase}{'_mni' if space == 'mni' else SCORED[source]}.nii.gz"
                    nib.save(nib.Nifti1Image(mask.astype(np.uint8), image.affine), walk_dir(later) / name)
            size = sizes[mm] = int(walk.sum())
            pick = order
            if phase == "60":
                # decision: the boost ball stays inside the base ball of the same distance, the NRG cone-down every other target follows
                pool = np.zeros(original.size, bool)
                # explainer: grown to the boost walk's size if resampling left it a few voxels past the base walk
                pool[balls_46[0][:max(balls_46[1][mm], size)]] = True
                pick = order[pool[order]]
            ball = np.zeros(original.size, bool)
            ball[pick[:size]] = True
            ball = ball.reshape(original.shape)
            for kind, target in (("walk", walk), ("iso", iso), ("ball", ball)):
                count = int(target.sum())
                inside_gtv = int(gtv.sum())
                row = {"patient": first[-8:-3], "pathology": measure.pathology(first), "phase": phase,
                       "walk_mm": mm, "target": kind, "volume_ml": round(count * ml, 1),
                       "beyond_gtv_ml": round(max(count - inside_gtv, 0) * ml, 1)}
                for code, mask in caught.items():
                    name = f"{code.replace('_beyond', '')}{'_beyond' if code.endswith('_beyond') else ''}_pct"
                    row[name] = round(100 * int((mask & target).sum()) / float(mask.sum()), 1) if mask.any() else None
                rows.append(row)
        if phase == "46":
            iso_46 = {mm: far <= mm for mm in DISTANCES}
            balls_46 = order, sizes
    return rows


def table(space, source, kind=""):
    # decision: the patient-tract walk gets its own tables so the atlas ones are never overwritten
    group = "" if COHORTS == ("UPenn",) else "_" + "_".join(c.lower() for c in COHORTS)
    return results_dir(space, RESCORED) / f"walkdown{SCORED[source]}{group}{'_covered' if COVERED else ''}{kind}.csv"


def has_tracts(first, source="dti_prep"):
    here = out_dir(first)
    return (here / TRACT_RUN[source] / "tracts_native.npz").exists() and (here / "midline" / "lateral.nii.gz").exists()


def pairs(space, source):
    found = measure.pairs(space, cohorts=COHORTS)
    # decision: a covered comparison keeps only patients with a preprocessed scan, the same set for atlas and patient tracts
    if source == "atlas" and not COVERED:
        return found
    return [(a, b) for a, b in found if has_tracts(a, "dti_prep" if source == "atlas" else source)]


def write(space, source="atlas"):
    rows = []
    for first, later in pairs(space, source):
        rows += score(first, later, space, source=source)
        measure.forget()
    for kind, keep in (("_curves", rows), ("", [r for r in rows if r["walk_mm"] == WALK_MM])):
        with open(table(space, source, kind), "w", newline="") as fh:
            out = csv.DictWriter(fh, fieldnames=list(keep[0]))
            out.writeheader()
            out.writerows(keep)
    return table(space, source, "_curves")


def _rows(space, source):
    with open(table(space, source)) as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        by.setdefault((r["phase"], r["target"]), {})[r["patient"]] = r
    return by


def _num(value):
    return None if value in ("", "None") else float(value)


def summarize(space, source="atlas"):
    from scipy.stats import wilcoxon
    by = _rows(space, source)
    path = table(space, source, "_summary")
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


def summarize_curves(space, source="atlas"):
    with open(table(space, source, "_curves")) as fh:
        rows = list(csv.DictReader(fh))
    cell = {}
    for r in rows:
        cell.setdefault((r["phase"], int(r["walk_mm"]), r["target"]), {})[r["patient"]] = r
    rng = np.random.default_rng(0)
    path = table(space, source, "_curves_summary")
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


def matched(space, source="atlas"):
    # decision: compare at equal coverage, nothing is traded so misses and brain never need weighting
    with open(table(space, source, "_curves")) as fh:
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
    path = table(space, source, "_matched")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    return path


def _build(job):
    first, source = job
    return first, build(first, source)


def main(source="atlas", *flags):
    from concurrent.futures import ProcessPoolExecutor
    global COVERED, COHORTS
    unknown = set(flags) - {"covered", "predict"}
    if unknown or ("predict" in flags and (source != "atlas" or "covered" in flags)):
        # explainer: PREDICT has no diffusion scans, so it has neither patient tracts nor a coverage map
        sys.exit(f"flags {flags}: use covered or predict, and predict only with atlas tracts")
    COVERED = "covered" in flags
    # decision: PREDICT here means TUM and LUMIERE, RHUH runs standalone and its PREDICT copy is left out
    COHORTS = ("TUM", "LUMIERE") if "predict" in flags else ("UPenn",)
    # decision: the patient-tract walk lives on the pre-op grid and coverage is a native map, so both score in the follow-up's own space only
    spaces = measure.SPACES if source == "atlas" and not COVERED else ("native",)
    firsts = [first for first, _ in pairs(spaces[-1], source)]
    # explainer: a walk already built from the same tracts is reused, rescoring never needs a rebuild
    todo = [first for first in firsts if not (walk_dir(first) / f"walk_allowed{BUILT[source]}.nii.gz").exists()]
    # decision: one thread per worker so they share the cores instead of fighting
    with ProcessPoolExecutor(8) as pool:
        for first, info in pool.map(_build, [(first, source) for first in todo]):
            print(first[-8:-3], info, flush=True)
    for space in spaces:
        print(write(space, source), summarize(space, source), summarize_curves(space, source),
              matched(space, source), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
