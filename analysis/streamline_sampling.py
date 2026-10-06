#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Does the share of the atlas we keep change the walk-down? The project keeps 1
# in 10 streamlines of every bundle. A convex hull is set by its outermost
# points, and more streamlines give more chances of an outlying end, so the
# share could be quietly setting the target's size. Here every patient's
# walk-down is rebuilt at 1/20, 1/10, 1/5, 1/2 and all of the atlas, each share
# inside the next (so they differ only in density), plus two more draws at 1/20
# and 1/10 to show how much a target moves just from which streamlines are
# drawn. MNI only, where the atlas is.

import csv
import multiprocessing
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure, walkdown
from analysis.capture_curves import FAINT, INK, PATIENT, save, style
from hcp import streamlines
from paths import results_dir

# (share kept, seed). Seed 0 is the nested series, 1/10 at seed 0 being exactly
# the share every other result uses
SHARES = ((1 / 20, 0), (1 / 10, 0), (1 / 5, 0), (1 / 2, 0), (1.0, 0),
          (1 / 20, 1), (1 / 20, 2), (1 / 10, 1), (1 / 10, 2))
DISTANCES = (10, 20, 40)
# the whole atlas held once and shared by forked workers; three at a time keeps
# the largest share well inside this laptop's memory
WORKERS = 3
# a worker never starts unless this much memory is free, in GB
SPARE_GB = 2.5

LINES = None


class Subset:
    # the same lookups walkdown.Lines gives, for any set of streamlines, kept
    # as views into one float32 array and flat voxel indices, so even the whole
    # atlas fits
    def __init__(self, data, keep, image, shape):
        points, offsets = data["points"], data["offsets"]
        self.lines = [points[offsets[i]:offsets[i + 1]] for i in keep]
        self.cut = np.cumsum([len(l) for l in self.lines])[:-1]
        inverse = np.linalg.inv(image.affine)
        flat, ok = [], []
        for line_chunk in np.array_split(np.arange(len(self.lines)), 50):
            stacked = np.vstack([self.lines[i] for i in line_chunk]).astype(float)
            ijk = np.rint(nib.affines.apply_affine(inverse, stacked)).astype(np.int32)
            good = np.all((ijk >= 0) & (ijk < shape), axis=1)
            ok.append(good)
            flat.append(np.ravel_multi_index(tuple(np.where(good[:, None], ijk, 0).T), shape)
                        .astype(np.int32))
        self.ok = np.concatenate(ok)
        self.flat = np.concatenate(flat)[self.ok]

    def lookup(self, mask):
        flags = np.zeros(len(self.ok), bool)
        flags[self.ok] = mask.ravel()[self.flat]
        return np.split(flags, self.cut)


def _one(pair):
    first, later = pair
    reach, allowed, _, info = walkdown.reach_mni(first, LINES)
    rows = [r for r in walkdown.score(first, later, "mni", reach=reach, allowed=allowed)
            if r["walk_mm"] in DISTANCES and r["target"] in ("walk", "ball")]
    measure.forget()
    return rows, info["46"]["exits"]


def _free_gb():
    with open("/proc/meminfo") as fh:
        info = dict(line.split(":", 1) for line in fh)
    return int(info["MemAvailable"].split()[0]) / 2 ** 20


def write():
    global LINES
    data = streamlines.everything()
    pairs = list(measure.pairs("mni"))
    seg, image = measure._seg(pairs[0][0], "mni")
    # each share is kept as it finishes, so a run cut short picks up where it
    # stopped; the parts go once the whole table is written
    parts = results_dir("mni") / "streamline_sampling_parts"
    parts.mkdir(exist_ok=True)
    out = []
    for share, seed in SHARES:
        part = parts / f"{round(1 / share)}_{seed}.csv"
        if part.exists():
            with open(part) as fh:
                out += list(csv.DictReader(fh))
            continue
        keep = streamlines.pick(data["bundle"], share, seed)
        LINES = Subset(data, keep, image, seg.shape)
        if _free_gb() < SPARE_GB * WORKERS:
            raise MemoryError(f"only {_free_gb():.1f} GB free; close something and rerun")
        rows = []
        # fork, so the workers share LINES rather than each building its own
        with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
            for got, exits in pool.imap(_one, pairs):
                for r in got:
                    r.update(share=f"1/{round(1 / share)}" if share < 1 else "all", seed=seed,
                             streamlines=len(keep), exits_46=exits)
                rows += got
        _save(part, rows)
        out += rows
        print(f"1/{1 / share:.0f} seed {seed}: {len(keep)} streamlines", flush=True)
        LINES = None
    path = results_dir("mni") / "streamline_sampling.csv"
    _save(path, out)
    for part in parts.iterdir():
        part.unlink()
    parts.rmdir()
    return path


def _save(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _table():
    with open(results_dir("mni") / "streamline_sampling.csv") as fh:
        rows = list(csv.DictReader(fh))
    num = lambda v: None if v in ("", "None") else float(v)
    # (share, seed, phase, mm, target) -> patient -> row
    by = {}
    for r in rows:
        key = (r["share"], int(r["seed"]), r["phase"], int(r["walk_mm"]), r["target"])
        by.setdefault(key, {})[r["patient"]] = {k: num(r[k]) for k in
                                                ("beyond_gtv_ml", "ET_pct", "WT_pct")}
    return by


NESTED = ("1/20", "1/10", "1/5", "1/2", "all")


def summarize():
    # per share and phase at each distance: median brain spent beyond the tumor,
    # median catch, the walk's median lead over the same-size ball, and how far
    # each patient's target sits from the whole-atlas one
    by = _table()
    out = []
    for share, seed in [(s, 0) for s in NESTED] + [("1/20", 1), ("1/20", 2), ("1/10", 1), ("1/10", 2)]:
        for phase, _ in walkdown.PHASES:
            for mm in DISTANCES:
                walk, ball = by[(share, seed, phase, mm, "walk")], by[(share, seed, phase, mm, "ball")]
                whole = by[("all", 0, phase, mm, "walk")]
                row = {"share": share, "seed": seed, "phase": phase, "walk_mm": mm,
                       "beyond_ml_median": np.median([r["beyond_gtv_ml"] for r in walk.values()])}
                change = [100 * (walk[p]["beyond_gtv_ml"] / whole[p]["beyond_gtv_ml"] - 1)
                          for p in walk if whole[p]["beyond_gtv_ml"]]
                row["volume_vs_all_pct_median"] = round(float(np.median(change)), 1)
                row["volume_vs_all_pct_max"] = round(float(np.max(np.abs(change))), 1)
                for code in ("ET", "WT"):
                    caught = [r[f"{code}_pct"] for r in walk.values() if r[f"{code}_pct"] is not None]
                    lead = [walk[p][f"{code}_pct"] - ball[p][f"{code}_pct"] for p in walk
                            if walk[p][f"{code}_pct"] is not None]
                    row[f"{code}_median"] = round(float(np.median(caught)), 1)
                    row[f"{code}_lead_over_ball_median"] = round(float(np.median(lead)), 2)
                out.append(row)
    path = results_dir("mni") / "streamline_sampling_summary.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    return path


def figure():
    # left: each patient's 46 Gy target at 20 mm against its whole-atlas size,
    # by share. Right: the walk's lead over the same-size ball on the whole
    # recurrence, by share. Grey = one patient, dark = median, open dots = the
    # repeat draws at 1/20 and 1/10
    by = _table()
    shares = list(range(len(NESTED)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    whole = by[("all", 0, "46", 20, "walk")]
    patients = sorted(whole)
    ax = axes[0]
    sizes = np.array([[by[(s, 0, "46", 20, "walk")][p]["beyond_gtv_ml"] / whole[p]["beyond_gtv_ml"] * 100
                       for s in NESTED] for p in patients if whole[p]["beyond_gtv_ml"]])
    for row in sizes:
        ax.plot(shares, row, color=PATIENT, lw=0.6, alpha=0.6)
    ax.plot(shares, np.median(sizes, axis=0), color=INK, lw=2.4, marker="o")
    for s, k in (("1/20", 0), ("1/10", 1)):
        for seed in (1, 2):
            again = [by[(s, seed, "46", 20, "walk")][p]["beyond_gtv_ml"] / whole[p]["beyond_gtv_ml"] * 100
                     for p in patients if whole[p]["beyond_gtv_ml"]]
            ax.plot(k, np.median(again), marker="o", mfc="none", color=INK, ms=8)
    ax.axhline(100, color=FAINT, lw=1, ls=(0, (4, 3)))
    ax.set_title("46 Gy walk-down at 20 mm: brain beyond the tumor,\n% of the whole-atlas target",
                 loc="left", fontsize=11, color=INK)
    ax = axes[1]
    lead = np.array([[by[(s, 0, "46", 20, "walk")][p]["WT_pct"] - by[(s, 0, "46", 20, "ball")][p]["WT_pct"]
                      for s in NESTED] for p in patients])
    for row in lead:
        ax.plot(shares, row, color=PATIENT, lw=0.6, alpha=0.6)
    ax.plot(shares, np.median(lead, axis=0), color=INK, lw=2.4, marker="o")
    ax.axhline(0, color=FAINT, lw=1, ls=(0, (4, 3)))
    span = np.percentile(np.abs(lead), 95)
    ax.set_ylim(-span, span)
    ax.set_title("46 Gy at 20 mm: whole recurrence caught, walk-down minus\nthe same-size ball "
                 "(percentage points)", loc="left", fontsize=11, color=INK)
    for ax in axes:
        ax.set_xticks(shares, [s.replace("all", "all\n480k") for s in NESTED])
        ax.set_xlabel("Share of each atlas bundle kept")
    fig.text(0.01, 0.005, "Grey = one patient, dark = median, open circles = medians of two more random "
             "draws at the same share. Each share holds every smaller one. MNI space.",
             color=FAINT, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.2, wspace=0.22)
    path = results_dir("mni", "figures") / "streamline_sampling.png"
    save(fig, path)
    return path


def main():
    style()
    if "--figure" not in sys.argv:
        print(write(), flush=True)
    print(summarize(), figure())


if __name__ == "__main__":
    main()
