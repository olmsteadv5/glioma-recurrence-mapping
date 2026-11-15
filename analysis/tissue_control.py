#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: how much of the walk-down's lead over a plain ball comes from simply covering more white matter

import csv
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from analysis.walkdown import RESCORED, SCORED, WALK_MM, has_tracts
from geometry import distance
from paths import THREADS, anatomy, out_dir, pair_dir, results_dir, stripped_dir, walk_dir

WM = (2, 41, 7, 46)


def _preop_tissue(first, later):
    # decision: tissue from the pre-op scan moved onto the follow-up grid, the follow-up's own labels already show where the tumor went
    dest = walk_dir(later) / "tissue_preop.nii.gz"
    if not dest.exists():
        subprocess.run(["greedy", "-d", "3", "-rf", str(anatomy(stripped_dir(later), later)), "-ri", "LABEL", "0.2vox",
                        "-rm", str(out_dir(first) / "tissue.nii.gz"), str(dest),
                        "-r", str(pair_dir(later) / "rigid_first_to_later.mat"), "-threads", THREADS],
                       capture_output=True, check=True)
    return np.asarray(nib.load(dest).dataobj)


def _nearest(order, pool, size):
    # explainer: the first size voxels of the pool in ball order
    ranked = order[pool.ravel()[order]]
    out = np.zeros(pool.size, bool)
    out[ranked[:size]] = True
    return out.reshape(pool.shape)


def one(first, later, source="atlas"):
    tag = SCORED[source]
    load = lambda name: np.asarray(nib.load(walk_dir(later) / f"{name}{tag}.nii.gz").dataobj)
    _, image = measure._seg(first, "native")
    gtv = measure.mask(first, "WT", "native")
    brain = measure._brain(first, "native")
    allowed = (load("walk_allowed") > 0) & brain
    far = np.where(allowed | gtv, np.maximum(distance(gtv, image), load("walk_gate_46")), np.inf)
    # decision: the same seeded random tie-break as the rescored walk-down
    order = np.lexsort((np.random.default_rng(int(first[-8:-3])).random(gtv.size), far.ravel()))
    wm = np.isin(_preop_tissue(first, later), WM)
    recurrence = measure.mask(later, "WT", "native")
    beyond_recurrence = recurrence & ~gtv
    target = load("walk_46") > 0
    size, beyond = int(target.sum()), target & ~gtv
    share = float(wm[beyond].mean()) if beyond.any() else 0.0
    reachable = np.isfinite(far)
    n_beyond = int(beyond.sum())
    n_wm = int(round(share * n_beyond))
    arms = {"target": target,
            "ball": _nearest(order, (allowed | gtv) & reachable, size),
            "white matter ball": _nearest(order, ((allowed & wm) | gtv) & reachable, size),
            # explainer: the same white matter share beyond the tumor as the target, both quotas filled nearest first
            "tissue matched ball": gtv | _nearest(order, allowed & wm & ~gtv & reachable, n_wm)
                                   | _nearest(order, allowed & ~wm & ~gtv & reachable, n_beyond - n_wm)}
    # explainer: a quota bigger than the nearby tissue leaves a ball smaller than the target, so sizes are kept to show it
    row = {"patient": first[-8:-3], "target_wm_share": round(share, 3),
           **{f"{name} voxels": int(mask.sum()) for name, mask in arms.items()},
           "recurrence_beyond_wm_share": round(float(wm[beyond_recurrence].mean()), 3) if beyond_recurrence.any() else None}
    for name, mask in arms.items():
        row[f"{name} WT"] = round(100 * float((recurrence & mask).sum()) / max(int(recurrence.sum()), 1), 2)
        row[f"{name} WT beyond"] = (round(100 * float((beyond_recurrence & mask).sum()) / int(beyond_recurrence.sum()), 2)
                                    if beyond_recurrence.any() else None)
    measure.forget()
    return row


def main(source="atlas", cohort=""):
    cohorts = ("TUM", "LUMIERE") if cohort == "predict" else ("UPenn",)
    found = measure.pairs("native", cohorts=cohorts)
    if source != "atlas":
        found = [(a, b) for a, b in found if has_tracts(a, source)]
    rows = [one(a, b, source) for a, b in found]
    path = results_dir("native", RESCORED) / f"tissue_control{SCORED[source]}{'_tum_lumiere' if cohort == 'predict' else ''}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path, f"walk {WALK_MM} mm, 46 Gy, {len(rows)} patients")
    for scope in ("WT", "WT beyond"):
        for control in ("ball", "white matter ball", "tissue matched ball"):
            d = np.array([r[f"target {scope}"] - r[f"{control} {scope}"] for r in rows
                          if r[f"target {scope}"] is not None and r[f"{control} {scope}"] is not None])
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            print(f"  {scope:10s} target minus {control:20s} {np.mean(d):+.2f} pts, better in {(d > 0.05).sum()}/{len(d)}, p={p:.2g}")


if __name__ == "__main__":
    main(*sys.argv[1:])
