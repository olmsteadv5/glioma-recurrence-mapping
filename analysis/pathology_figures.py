#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Where the part of the later tumor a 2 cm margin misses ends up, and how much
# brain the margin spends compared with a plan that knew where the tumor would
# return, both split by the pathology of the second resection.

import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from analysis.capture_curves import FAINT, INK, PATIENT, SOFT, capture, save, style
from geometry import MARGIN_MM, ml_per_voxel
from paths import MNI, OUTPUTS

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#c8323c"
PARTS = (("ET", "Enhancing"), ("WT", "Whole tumor"))
# the dataset's six pathology scores, as the share of tissue with malignant features
SCORES = ((6, "over 90%"), (5, "75-90%"), (4, "50-75%"), (3, "25-50%"), (2, "10-25%"), (1, "under 10%"))
# three groups for the curves, since one score alone is as few as 3 patients
GROUPS = (("Pathology 5-6", (5, 6)), ("Pathology 3-4", (3, 4)), ("Pathology 1-2", (1, 2)))
# why: Lee et al.'s central class, 95% of the recurrence inside the field [lee99]
LEVEL = 0.95
MM = np.arange(0, 90.25, 0.25)
VOLUME = np.arange(0, 601, 1.0)


def tint(color, amount):
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


def collect():
    rows = []
    for first, later in measure.pairs():
        row = {"patient": first[-8:-3], "days": measure.interval(first),
               "score": measure.pathology(first), "extra": measure.margin_ml(first, MM),
               "extra_20": float(measure.margin_ml(first, MARGIN_MM))}
        for code, _ in PARTS:
            far = measure.reach(first, later, code)
            row[code] = None if far is None else {"far": far, "zones": measure.zones(first, later, code)}
        rows.append(row)
        measure.forget()
    return rows


def failure_pattern(rows, path):
    zones = (("inside", "Inside original tumor", tint(BLUE, 0.55)), ("within", "Outside, within 20 mm", BLUE),
             ("beyond", "Beyond 20 mm", ORANGE), ("separate", "Separate lesion beyond 20 mm", RED))
    groups = [sorted((r for r in rows if r["score"] == s and r["WT"]), key=lambda r: r["days"]) for s, _ in SCORES]
    heights = [len(g) + 1.2 for g in groups]
    fig, axes = plt.subplots(len(SCORES), 2, figsize=(13, 0.26 * sum(heights) + 4.4),
                             gridspec_kw={"height_ratios": heights}, sharex=True)
    for row, ((score, share), group) in enumerate(zip(SCORES, groups)):
        for col, (code, part) in enumerate(PARTS):
            ax = axes[row, col]
            for i, r in enumerate(group):
                if r[code] is None:
                    ax.text(1, i, "no enhancing tumor", va="center", fontsize=8, color=FAINT)
                    continue
                z = r[code]["zones"]
                total, left = sum(z.values()), 0
                for key, _, color in zones:
                    ax.barh(i, 100 * z[key] / total, left=left, color=color, height=0.72, lw=0)
                    left += 100 * z[key] / total
                out = z["beyond"] + z["separate"]
                label = f"{total:.1f} mL, {out:.1f} beyond" if out >= 0.05 else f"{total:.1f} mL"
                ax.text(101.5, i, label, va="center", fontsize=8, color=SOFT)
            ax.set_yticks(range(len(group)))
            ax.set_yticklabels([f"{r['patient']}  {r['days']} d" for r in group] if col == 0 else [], fontsize=8)
            ax.set_ylim(len(group) - 0.4, -0.6)
            ax.set_xlim(0, 100)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.set_title(f"{part} ({sum(r[code] is not None for r in group)})", loc="left", fontsize=11)
            if col == 0:
                ax.text(-0.2, 1 + 1.9 / heights[row], f"Pathology {score} ({share} malignant)",
                        transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, va="bottom")
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlabel("Share of the follow-up tumor")
    fig.legend([Patch(color=c) for *_, c in zones], [name for _, name, _ in zones], loc="upper left",
               bbox_to_anchor=(0.08, 1.01), ncol=4, frameon=False, handlelength=1.1, columnspacing=1.6)
    fig.text(0.01, 0.005, "One row per patient, sorted by days between scans. Distances are from the edge of "
             "the whole original tumor.\nRight of each bar: follow-up volume of that part, and how much of it "
             "lies beyond 20 mm (shown when at least 0.1 mL).", color=FAINT, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.955, bottom=0.05, hspace=0.5, wspace=0.36)
    save(fig, path)


def curves(rows, code, ml):
    uniform, best, points = [], [], []
    for r in rows:
        if r[code] is None:
            continue
        far = r[code]["far"]
        uniform.append(np.interp(VOLUME, r["extra"], capture(far, MM)))
        # the best plan adds only brain the tumor came back in, nearest first
        inside, outside = int((far == 0).sum()), int((np.isfinite(far) & (far > 0)).sum())
        best.append((inside + np.minimum(VOLUME / ml, outside)) / far.size)
        points.append((r["extra_20"], capture(far, MARGIN_MM)))
    return 100 * np.array(uniform), 100 * np.array(best), np.array(points)


def crossing(curve, level):
    i = int(np.argmax(curve >= level))
    if curve[i] < level:
        return None
    return float(VOLUME[i]) if i == 0 else float(np.interp(level, curve[i - 1:i + 1], VOLUME[i - 1:i + 1]))


def headroom(rows, path):
    ml = ml_per_voxel(nib.load(MNI))
    level = 100 * LEVEL
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(11, 7), sharex=True, sharey=True)
    for i, (code, part) in enumerate(PARTS):
        for j, (name, scores) in enumerate(GROUPS):
            ax = axes[i, j]
            uniform, best, points = curves([r for r in rows if r["score"] in scores], code, ml)
            mid, top = np.median(uniform, axis=0), np.median(best, axis=0)
            ax.fill_between(VOLUME, *np.percentile(uniform, [25, 75], axis=0), color=tint(BLUE, 0.72), lw=0, zorder=2)
            ax.plot(VOLUME, mid, color=BLUE, lw=2.2, zorder=4)
            ax.fill_between(VOLUME, *np.percentile(best, [25, 75], axis=0), color=tint(SOFT, 0.85), lw=0, zorder=1)
            ax.plot(VOLUME, top, color=SOFT, lw=1.2, ls=(0, (4, 2)), zorder=3)
            ax.scatter(points[:, 0], 100 * points[:, 1], s=10, color=PATIENT, lw=0, zorder=2.5)
            at = float(np.median(points[:, 0]))
            here = float(np.interp(at, VOLUME, mid))
            ax.plot(at, here, "o", ms=6, mfc="white", mec=BLUE, mew=1.4, zorder=6, clip_on=False)
            ax.annotate(f"2 cm\n{at:.0f} mL, {here:.0f}%", (at, here), xytext=(0, -20), textcoords="offset points",
                        fontsize=8, ha="center", va="top")
            a, b = crossing(top, level), crossing(mid, level)
            if a is not None and b is not None:
                ax.plot([a, b], [level, level], color=ORANGE, lw=1.8, marker="|", ms=7, mew=1.8, zorder=5)
                ax.text(b + 8, level - 4, f"{b - a:.0f} mL", color=ORANGE, fontsize=9, va="top", zorder=6)
            ax.set_xlim(0, VOLUME[-1])
            ax.set_ylim(0, 101.5)
            ax.yaxis.set_major_locator(MultipleLocator(25))
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.xaxis.set_major_locator(MultipleLocator(200))
            ax.set_title(f"{part} ({len(uniform)})", loc="left", fontsize=11)
            if i == 0:
                ax.text(0.5, 1.2, name, transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold")
    fig.supxlabel("Brain irradiated beyond the original tumor (mL)", y=0.06, fontsize=10, color=SOFT)
    fig.supylabel("Recurrence captured (% of volume)", x=0.02, fontsize=10, color=SOFT)
    fig.text(0.02, 0.005, "Blue line = uniform expansion, median. Dashed line = best possible, median. Shaded band "
             "= middle 50 percentile.\nBest possible = expanding only into the voxels where the tumor came back. "
             f"Grey dot = patient at 2 cm. Orange = volume saved at {level:.0f}% capture.", color=FAINT, fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.87, bottom=0.16, wspace=0.12, hspace=0.45)
    save(fig, path)


def main():
    figures = OUTPUTS / "figures"
    figures.mkdir(exist_ok=True)
    rows = collect()
    style()
    failure_pattern(rows, figures / "failure_pattern.png")
    headroom(rows, figures / "headroom.png")


if __name__ == "__main__":
    main()
