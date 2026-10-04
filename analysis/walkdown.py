#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# A target that follows white matter instead of growing evenly: walk 20 mm along
# every atlas tract that leaves the tumor, then wrap one convex hull around the
# tumor and where the walks ended. Built for both NRG phases, in MNI where the
# tracts are, and pulled into each follow-up scan for patient space.

import csv
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from geometry import MARGIN_MM, distance, ml_per_voxel, outline, template_brain
from paths import THREADS, modality, out_dir, results_dir, stripped_dir
from show.compare import _inverse
from yale import tracts

# why: the NRG margin, 2 cm along the tract instead of in every direction [rtog0825]
WALK_MM = MARGIN_MM
# why: NRG lets the margin shrink to 0.5 cm at barriers, so no target goes below it [rtog0825]
FLOOR_MM = 5
# a tract that leaves and comes back within this is running along the edge,
# not leaving. A judgement call, about three voxels; not from any paper
SKIM_MM = 3
# why: the falx is a hard barrier unless the FLAIR already crosses the callosum [nrg19]
CROSSED_ML = 1.0

# the two NRG phases: the 46 Gy volume grows from the whole abnormality, the
# 60 Gy boost from enhancement and cavity, which on a pre-op scan is the core
PHASES = (("46", "WT"), ("60", "TC"))


class Lines:
    # every centerline point's voxel on the MNI grid, found once: every patient's
    # MNI masks share that grid, so a mask lookup is then a single index
    def __init__(self, image, shape):
        self.lines = tracts.centerlines()
        stacked = np.vstack(self.lines)
        self.cut = np.cumsum([len(l) for l in self.lines])[:-1]
        ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(image.affine), stacked)).astype(int)
        self.ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
        self.ijk = tuple(ijk[self.ok].T)

    def lookup(self, mask):
        flags = np.zeros(len(self.ok), bool)
        flags[self.ok] = mask[self.ijk]
        return np.split(flags, self.cut)


def _walk(line, inside, allowed):
    # one end of the walk for every place this line leaves the tumor, in both
    # directions along it. It stops at 20 mm of path, at the line's end, at the
    # edge of where a plan may go, or where it comes back into the tumor
    ends = []
    for points, gtv, ok in ((line, inside, allowed), (line[::-1], inside[::-1], allowed[::-1])):
        step = np.linalg.norm(np.diff(points, axis=0), axis=1)
        for i in np.flatnonzero(gtv[:-1] & ~gtv[1:]):
            back = np.flatnonzero(gtv[i + 1:])
            if back.size and step[i:i + 1 + back[0]].sum() < SKIM_MM:
                continue
            walked, j = 0.0, i
            while j + 1 < len(points) and ok[j + 1] and not gtv[j + 1]:
                if walked + step[j] >= WALK_MM:
                    t = (WALK_MM - walked) / step[j]
                    ends.append(points[j] + t * (points[j + 1] - points[j]))
                    break
                walked += step[j]
                j += 1
            else:
                ends.append(points[j])
    return ends


def _hull(points, image, shape):
    # every voxel whose center is inside the convex hull of the points
    hull = ConvexHull(points)
    inverse = np.linalg.inv(image.affine)
    corners = nib.affines.apply_affine(inverse, points[hull.vertices])
    low = np.maximum(np.floor(corners.min(0)).astype(int), 0)
    high = np.minimum(np.ceil(corners.max(0)).astype(int) + 1, shape)
    box = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(low, high)], indexing="ij"), -1)
    box = box.reshape(-1, 3)
    out = np.zeros(shape, bool)
    for start in range(0, len(box), 500_000):
        chunk = box[start:start + 500_000]
        world = nib.affines.apply_affine(image.affine, chunk)
        keep = np.all(world @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-6, axis=1)
        out[tuple(chunk[keep].T)] = True
    return out


def allowed_mni(first):
    # the brain, minus the other hemisphere unless the tumor already crosses
    data, image = measure._seg(first, "mni")
    brain = template_brain()
    x = nib.affines.apply_affine(image.affine, np.indices(data.shape).reshape(3, -1).T)[:, 0]
    left = (x < 0).reshape(data.shape)
    tumor = data > 0
    ml = ml_per_voxel(image)
    small = min((tumor & left).sum(), (tumor & ~left).sum()) * ml
    if small >= CROSSED_ML:
        return brain, True
    side = left if (tumor & left).sum() > (tumor & ~left).sum() else ~left
    return brain & side, False


def build_mni(first, lines):
    # both walk targets in MNI, plus the endpoints, which the viewer can draw
    data, image = measure._seg(first, "mni")
    allowed, crossed = allowed_mni(first)
    may = lines.lookup(allowed)
    built, info = {}, {"crossed": crossed}
    for phase, region in PHASES:
        gtv = measure.mask(first, region, "mni")
        if not gtv.any():
            gtv = data > 0
        ends, used = [], 0
        for line, inside, ok in zip(lines.lines, lines.lookup(gtv), may):
            if inside.any():
                used += 1
                ends += _walk(line, inside, ok)
        edge = nib.affines.apply_affine(image.affine, np.argwhere(outline(gtv)))
        cloud = np.vstack([edge] + ([np.array(ends)] if ends else []))
        target = _hull(cloud, image, gtv.shape) & allowed
        target |= (distance(gtv, image) <= FLOOR_MM) & allowed
        target |= gtv
        if phase == "60":
            target &= built["46"] | gtv
        built[phase] = target
        info[phase] = {"tracts": used, "ends": len(ends)}
        np.save(out_dir(first) / f"walk_{phase}_ends.npy", np.array(ends).reshape(-1, 3))
        nib.save(nib.Nifti1Image(target.astype(np.uint8), image.affine),
                 out_dir(first) / f"walk_{phase}_mni.nii.gz")
    nib.save(nib.Nifti1Image(allowed.astype(np.uint8), image.affine),
             out_dir(first) / "walk_allowed_mni.nii.gz")
    return info


def to_native(first, later, name):
    # MNI -> the pre-op scan -> the follow-up scan, the same road the pre-op tumor
    # takes into patient space, so the two line up
    source = out_dir(first) / f"{name}_mni.nii.gz"
    dest = out_dir(later) / f"{name}.nii.gz"
    subprocess.run(["greedy", "-d", "3", "-rf", str(modality(stripped_dir(later), later, "T1")),
                    "-ri", "LABEL", "0.2vox", "-rm", str(source), str(dest),
                    "-r", str(out_dir(later) / "rigid_first_to_later.mat"),
                    str(out_dir(first) / "affine.mat") + ",-1", str(_inverse(first)),
                    "-threads", THREADS], capture_output=True, check=True)
    return np.asarray(nib.load(dest).dataobj) > 0


def targets(first, later, space):
    # every target this comparison scores, in one space
    if space == "mni":
        load = lambda name: np.asarray(nib.load(out_dir(first) / f"{name}_mni.nii.gz").dataobj) > 0
    else:
        load = lambda name: to_native(first, later, name)
    allowed = load("walk_allowed") & measure._brain(first, space)
    _, image = measure._seg(first, space)
    out = {}
    for phase, region in PHASES:
        gtv = measure.mask(first, region, space)
        if not gtv.any():
            gtv = measure.mask(first, "WT", space)
        walk = (load(f"walk_{phase}") & allowed) | gtv
        far = np.where(allowed | gtv, distance(gtv, image), np.inf)
        iso = far <= MARGIN_MM
        # the same number of voxels as the walk, nearest the tumor first
        order = np.argsort(far, axis=None, kind="stable")[:int(walk.sum())]
        ball = np.zeros(gtv.shape, bool)
        ball.flat[order] = True
        if phase == "60":
            iso &= out["iso_46"]
        out[f"walk_{phase}"], out[f"iso_{phase}"], out[f"ball_{phase}"] = walk, iso, ball
        out[f"gtv_{phase}"] = gtv
    return out


def score(first, later, space):
    built = targets(first, later, space)
    ml = measure.voxel_ml(first, space)
    rows = []
    for phase, _ in PHASES:
        gtv = built[f"gtv_{phase}"]
        for kind in ("walk", "iso", "ball"):
            target = built[f"{kind}_{phase}"]
            row = {"patient": first[-8:-3], "pathology": measure.pathology(first), "phase": phase,
                   "target": kind, "volume_ml": round(float(target.sum()) * ml, 1),
                   "beyond_gtv_ml": round(float((target & ~gtv).sum()) * ml, 1)}
            for code in ("ET", "WT"):
                now = measure.mask(later, code, space)
                row[f"{code}_pct"] = (round(100 * float((now & target).sum()) / float(now.sum()), 1)
                                      if now.any() else None)
            rows.append(row)
    return rows


def write(space):
    path = results_dir(space) / "walkdown.csv"
    rows = []
    for first, later in measure.pairs(space):
        rows += score(first, later, space)
        measure.forget()
    with open(path, "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    return path


LABEL = {"iso": "NRG 2 cm", "walk": "Walk-down", "ball": "Same-volume ball"}
PART = {"ET": "Enhancing", "WT": "Whole tumor"}


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


def figure(space):
    # each patient's walk-down against the ball of the same volume, and against
    # the NRG 2 cm volume, with what each costs in brain
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.capture_curves import FAINT, INK, PATIENT, SOFT, save, style
    style()
    by = _rows(space)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))
    for i, (phase, _) in enumerate(PHASES):
        walk, ball, iso = (by[(phase, k)] for k in ("walk", "ball", "iso"))
        title = "46 Gy volume, from the whole tumor" if phase == "46" else "60 Gy boost, from the tumor core"
        for j, code in enumerate(("ET", "WT")):
            ax = axes[i, j]
            pts = [(_num(ball[p][f"{code}_pct"]), _num(walk[p][f"{code}_pct"])) for p in walk
                   if _num(walk[p][f"{code}_pct"]) is not None]
            x, y = np.array(pts).T
            low = max(0, min(x.min(), y.min()) - 3)
            ax.plot([low, 100.5], [low, 100.5], color=FAINT, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.scatter(x, y, s=20, color="#2a78d6", lw=0, zorder=3)
            ax.set_xlim(low, 101)
            ax.set_ylim(low, 101)
            ax.set_aspect("equal")
            ax.set_xlabel("Same-volume ball (% captured)")
            ax.set_ylabel("Walk-down (% captured)")
            ax.set_title(f"{PART[code]} recurrence ({len(x)})", loc="left", fontsize=11)
            ax.text(0.97, 0.04, f"walk better {int((y - x > 0.05).sum())}, worse {int((x - y > 0.05).sum())}",
                    transform=ax.transAxes, ha="right", fontsize=9, color=SOFT)
        ax = axes[i, 2]
        for k, kind in enumerate(("iso", "walk")):
            group = by[(phase, kind)]
            vals = [_num(r["beyond_gtv_ml"]) for r in group.values()]
            ax.boxplot([vals], positions=[k], widths=0.5, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": "white", "edgecolor": FAINT},
                       whiskerprops={"color": FAINT}, capprops={"color": FAINT},
                       medianprops={"color": "#eb6834", "lw": 2.2})
            rng = np.random.default_rng(0)
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(vals)), vals, s=10, color=PATIENT, lw=0, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([LABEL["iso"], LABEL["walk"]])
        ax.set_ylabel("Brain beyond the tumor (mL)")
        ax.set_ylim(bottom=0)
        ax.set_title("Cost", loc="left", fontsize=11)
        fig.text(0.01, 0.975 - i * 0.49, title, fontsize=12, color=INK, weight="bold")
    fig.text(0.01, 0.005, "Dot = patient. Dashed line = no difference; above it the walk-down "
             "catches more than a ball of exactly its size.\nWalk-down = convex hull around the "
             "tumor and the points 20 mm along every atlas tract leaving it. Box = middle 50 "
             "percentile, line = median.", color=FAINT, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.1, hspace=0.42, wspace=0.28)
    path = results_dir(space) / "walkdown.png"
    save(fig, path)
    return path


def main():
    lines = None
    for first, later in measure.pairs("mni"):
        if lines is None:
            data, image = measure._seg(first, "mni")
            lines = Lines(image, data.shape)
        info = build_mni(first, lines)
        print(first[-8:-3], info, flush=True)
        measure.forget()
    for space in measure.SPACES:
        print(write(space), summarize(space), figure(space))


if __name__ == "__main__":
    main()
