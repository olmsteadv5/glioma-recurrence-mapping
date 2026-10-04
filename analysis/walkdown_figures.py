#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# The walk-down figures for sending: what each target costs and catches at every
# walk distance from 0 to 40 mm, next to a plain margin grown the same distance
# and a ball of exactly the walk-down's size, and the brain the walk-down saves
# once it catches as much as the NRG 2 cm volume. Run walkdown.py first.

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from analysis.capture_curves import FAINT, INK, save, style
from paths import results_dir

# drawn in this order, so the walk-down sits on top of the ball it matches in size
TARGETS = (("iso", "Plain margin, same distance", "#8f6aa8"),
           ("ball", "Ball of the walk-down's size", "#eb6834"),
           ("walk", "Walk-down", "#2a78d6"))
COLOR = {key: color for key, _, color in TARGETS}
PHASES = (("46", "46 Gy volume (from the whole tumor)"), ("60", "60 Gy boost (from the tumor core)"))
MEASURES = (("beyond_gtv_ml", "Brain irradiated beyond the tumor (mL)"),
            ("ET_pct", "Enhancing recurrence caught (%)"),
            ("WT_pct", "Whole-tumor recurrence caught (%)"))
LABELED = (0, 10, 20, 30, 40)
# the NRG margin, marked on every distance axis
NRG_MM = 20


def load(space):
    # per phase and target, one array per measure: rows are patients, columns mm
    with open(results_dir(space) / "walkdown_curves.csv") as fh:
        rows = list(csv.DictReader(fh))
    distances = sorted({int(r["walk_mm"]) for r in rows})
    patients = sorted({r["patient"] for r in rows})
    out = {}
    for r in rows:
        grid = out.setdefault((r["phase"], r["target"]), {
            key: np.full((len(patients), len(distances)), np.nan) for key, _ in MEASURES})
        for key, _ in MEASURES:
            if r[key] not in ("", "None"):
                grid[key][patients.index(r["patient"]), distances.index(int(r["walk_mm"]))] = float(r[key])
    return out, np.array(distances)


def summary(space):
    with open(results_dir(space) / "walkdown_curves_summary.csv") as fh:
        return [r for r in csv.DictReader(fh) if r["target"] == "ball"]


def reaches(curve, distances, share=95):
    # the first distance at which a median curve gets to share, or None
    hit = np.flatnonzero(curve >= share)
    return int(distances[hit[0]]) if hit.size else None


def end_labels(ax, ends, unit):
    # the value of each curve at 20 mm, nudged apart so none sit on another
    span = np.diff(ax.get_ylim())[0]
    placed = []
    for value, color in sorted(ends):
        y = value if not placed else max(value, placed[-1] + span * 0.055)
        placed.append(y)
        shown = f"{value:.0f} mL" if unit == "mL" else f"{value:.1f}%"
        ax.annotate(shown, (1.0, y), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=color, weight="bold")


def by_distance(by, distances, path):
    # one row per phase, each measure against walk distance, median and middle half
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8))
    for i, (phase, title) in enumerate(PHASES):
        for j, (key, label) in enumerate(MEASURES):
            ax = axes[i, j]
            ends = []
            for kind, name, color in TARGETS:
                grid = by[(phase, kind)][key]
                low, mid, high = np.nanpercentile(grid, [25, 50, 75], axis=0)
                ax.fill_between(distances, low, high, color=color, alpha=0.12, lw=0)
                same = kind == "walk" and key == "beyond_gtv_ml"
                ax.plot(distances, mid, color=color, lw=2.2, label=name, ls=(0, (5, 3)) if same else "-")
                # the walk-down and its ball are one size by construction, so one label
                if not same:
                    ends.append((mid[-1], color))
                if key == "ET_pct":
                    at = reaches(mid, distances)
                    if at is not None:
                        ax.plot(at, mid[distances == at][0], "o", ms=6, mfc="white", mec=color,
                                mew=1.6, zorder=4)
            ax.set_xlim(0, distances[-1])
            ax.set_xticks(range(0, distances[-1] + 1, 10))
            ax.axvline(NRG_MM, color=FAINT, lw=1, ls=(0, (4, 3)), zorder=0)
            ax.set_xlabel("Walk distance (mm)")
            if key != "beyond_gtv_ml":
                ax.set_ylim(0, 102)
            else:
                ax.set_ylim(bottom=0)
            end_labels(ax, ends, "mL" if key == "beyond_gtv_ml" else "%")
            ax.set_title(label, loc="left", fontsize=11, color=INK)
            if j == 0:
                ax.text(-0.2, 1.16, title, transform=ax.transAxes, fontsize=12.5, weight="bold", color=INK)
    axes[0, 2].legend(loc="lower right", frameon=False, fontsize=9)
    fig.text(0.01, 0.005, "Line = median patient, band = middle 50 percentile, sampled every 1 mm; dashed = 20 mm, the NRG margin. "
             "Open circle = where the median first catches 95% of the enhancing recurrence. The ball\n"
             "is the walk-down's size by construction, so their volume lines coincide (dashed blue). "
             "Walk-down = convex hull around the pre-op tumor and the points d mm along\nevery atlas "
             "tract leaving it, no minimum margin. All targets trimmed at the brain edge and the "
             "midline (NRG rule). Patient space, BraTS 2025 winning segmentation.", color=FAINT, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.9, bottom=0.13, hspace=0.6, wspace=0.32)
    save(fig, path)


def against_volume(by, distances, rows, path):
    # the fair comparison: catch against brain spent, and walk-down minus the
    # same-size ball at each distance
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8))
    for i, (phase, title) in enumerate(PHASES):
        for j, (key, label) in enumerate(MEASURES[1:]):
            ax = axes[i, j]
            for kind in ("iso", "ball", "walk"):
                # the ball is placed at the walk-down's own volume, which it equals
                # patient by patient, so blue against orange is a like-for-like gap
                size = "iso" if kind == "iso" else "walk"
                spend = np.nanmedian(by[(phase, size)]["beyond_gtv_ml"], axis=0)
                caught = np.nanmedian(by[(phase, kind)][key], axis=0)
                ax.plot(spend, caught, color=COLOR[kind], lw=2, marker="o", ms=2.5)
                if kind == "ball":
                    continue
                for mm in LABELED:
                    k = int(np.flatnonzero(distances == mm)[0])
                    ax.annotate(f"{mm}", (spend[k], caught[k]), xytext=(3, -9 if kind == "iso" else 4),
                                textcoords="offset points", fontsize=8, color=COLOR[kind])
                if kind == "iso":
                    k = int(np.flatnonzero(distances == NRG_MM)[0])
                    ax.plot(spend[k], caught[k], "*", ms=13, color=COLOR["iso"], mec="white", zorder=5)
                    ax.annotate("NRG 2 cm", (spend[k], caught[k]), xytext=(6, -14),
                                textcoords="offset points", fontsize=9, color=COLOR["iso"], weight="bold")
            ax.set_xlim(left=0)
            ax.set_ylim(0, 102)
            ax.set_xlabel("Brain irradiated beyond the tumor, median (mL)")
            ax.set_title(label.replace(" (%)", ", median (%)"), loc="left", fontsize=11, color=INK)
            if j == 0:
                ax.text(-0.2, 1.16, title, transform=ax.transAxes, fontsize=12.5, weight="bold", color=INK)
        ax = axes[i, 2]
        for code, tone in (("WT", "#2a78d6"), ("ET", "#7a8a99")):
            mine = [r for r in rows if r["phase"] == phase]
            d = np.array([int(r["walk_mm"]) for r in mine])
            mid = np.array([float(r[f"{code}_walk_minus_ball"]) for r in mine])
            lo = np.array([float(r[f"{code}_low"]) for r in mine])
            hi = np.array([float(r[f"{code}_high"]) for r in mine])
            ax.fill_between(d, lo, hi, color=tone, alpha=0.15, lw=0)
            ax.plot(d, mid, color=tone, lw=2.2, label="whole tumor" if code == "WT" else "enhancing")
        ax.axhline(0, color=FAINT, lw=1, ls=(0, (4, 3)))
        ax.axvline(NRG_MM, color=FAINT, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.set_xlim(0, distances[-1])
        ax.set_xticks(range(0, distances[-1] + 1, 10))
        ax.set_xlabel("Walk distance (mm)")
        ax.set_title("Walk-down minus same-size ball,\nmedian (percentage points)", loc="left",
                     fontsize=11, color=INK)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.text(0.01, 0.005, "Left and middle: each dot is one walk distance (numbers mark 0, 10, 20, 30, 40 mm; star = the NRG 2 cm volume). "
             "Blue = walk-down, orange = a ball of exactly its size in each patient, purple =\n"
             "plain margin grown the same distance. Blue above orange means the tracts caught more "
             "for the same brain. Right: the median per-patient walk-down minus ball, band =\n"
             "bootstrap 95% interval; above zero the walk-down's shape beats its size. Patient space.",
             color=FAINT, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.14, hspace=0.62, wspace=0.3)
    save(fig, path)


def equal_coverage(space, path):
    # per patient: how much brain the walk-down saves once it catches as much as
    # the NRG 2 cm volume, so nothing is traded and no weighting is needed
    with open(results_dir(space) / "walkdown_matched.csv") as fh:
        rows = list(csv.DictReader(fh))
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (phase, code) in zip(axes, [(p, c) for p, _ in PHASES for c in ("ET", "WT")]):
        mine = [r for r in rows if r["phase"] == phase and r["recurrence"] == code]
        got = np.array([float(r["saved_ml"]) for r in mine if r["saved_ml"] not in ("", "None")])
        needed = [int(r["walk_mm_needed"]) for r in mine if r["walk_mm_needed"] not in ("", "None")]
        ax.axhline(0, color=FAINT, lw=1, ls=(0, (4, 3)))
        ax.scatter(rng.uniform(-0.15, 0.15, len(got)), got, s=20,
                   color=np.where(got >= 0, COLOR["walk"], COLOR["ball"]), lw=0, zorder=3)
        if len(got):
            ax.plot([-0.28, 0.28], [np.median(got)] * 2, color=INK, lw=2.4, zorder=4)
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        part = "Enhancing" if code == "ET" else "Whole tumor"
        ax.set_title(f"{phase} Gy, {part.lower()}", loc="left", fontsize=11, color=INK)
        lines = [f"reach 2 cm's coverage: {len(got)} of {len(mine)}"]
        if len(got):
            lines += [f"walk needed: median {np.median(needed):.0f} mm",
                      f"brain saved: median {np.median(got):.0f} mL",
                      f"saved in {int((got > 0).sum())}, cost more in {int((got < 0).sum())}"]
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=9, color=INK,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none", "alpha": 0.9}, zorder=5)
    axes[0].set_ylabel("Brain saved vs NRG 2 cm at equal coverage (mL)")
    top = axes[0].get_ylim()[1]
    axes[0].set_ylim(top=top * 1.3)
    fig.text(0.01, 0.005, "Dot = patient whose walk-down, walked 0 to 40 mm, catches at least as much "
             "recurrence as the NRG 2 cm volume; the shortest such walk is used. Blue = it needed less\n"
             "brain, orange = more. Bar = median. Patients whose walk-down never matches the 2 cm "
             "volume within 40 mm are left out and counted above. Patient space.", color=FAINT, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.16, wspace=0.08)
    save(fig, path)


def main():
    style()
    for space in measure.SPACES:
        by, distances = load(space)
        figures = results_dir(space, "figures")
        by_distance(by, distances, figures / "walkdown_by_distance.png")
        against_volume(by, distances, summary(space), figures / "walkdown_vs_volume.png")
        equal_coverage(space, figures / "walkdown_equal_coverage.png")
        (figures / "walkdown_compare.png").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
