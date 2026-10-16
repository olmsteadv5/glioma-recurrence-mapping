#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# How much of the recurrence lands on the original tumor, against its size and
# the pathology of the second resection, in both spaces.

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from analysis.capture_curves import FAINT, PATIENT, SOFT, capture, save, style
from paths import results_dir

BLUE, ORANGE = "#2a78d6", "#eb6834"
PARTS = (("ET", "Enhancing"), ("WT", "Whole tumor"))
MARGINS = (0, 5, 10, 15, 20)
SCORES = (1, 2, 3, 4, 5, 6)
NOTE = "Pathology score = share of the second resection with malignant features, 1 = under 10%, 6 = over 90%."


def collect(space):
    rows = []
    for first, later in measure.pairs(space):
        ml = measure.voxel_ml(first, space)
        row = {"score": measure.pathology(first)}
        for code, _ in PARTS:
            row[f"first_{code}"] = float(measure.mask(first, code, space).sum()) * ml
            far = measure.reach(first, later, code, space)
            row[code] = None if far is None else {"ml": far.size * ml,
                                                  "pct": {mm: 100 * capture(far, mm) for mm in MARGINS}}
        rows.append(row)
        measure.forget()
    return rows


def boxes(ax, groups, color):
    # boxes for the spread, every patient on top since some scores hold 3
    keep = [i for i, g in enumerate(groups) if len(g)]
    ax.boxplot([groups[i] for i in keep], positions=[SCORES[i] for i in keep], widths=0.55,
               showfliers=False, patch_artist=True,
               boxprops={"facecolor": "white", "edgecolor": FAINT},
               whiskerprops={"color": FAINT}, capprops={"color": FAINT},
               medianprops={"color": color, "lw": 2.2})
    rng = np.random.default_rng(0)
    for i in keep:
        ax.scatter(SCORES[i] + rng.uniform(-0.14, 0.14, len(groups[i])), groups[i],
                   s=12, color=PATIENT, lw=0, zorder=3)
    ax.set_xticks(SCORES)
    ax.set_xlim(0.4, 6.6)


def overlap_vs_size(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (code, part) in zip(axes, PARTS):
        have = [r for r in rows if r[code]]
        x = np.array([r["first_WT"] for r in have])
        y = np.array([r[code]["pct"][0] for r in have])
        ax.scatter(x, y, s=22, color=BLUE, lw=0)
        rho, p = spearmanr(x, y)
        shown = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.98, 0.04, f"Spearman ρ = {rho:.2f}, {shown}", transform=ax.transAxes,
                ha="right", fontsize=9, color=SOFT)
        ax.set_title(f"{part} ({len(have)})", loc="left", fontsize=11)
        ax.set_xlabel("Original tumor volume (mL)")
        ax.set_ylim(0, 102)
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axes[0].set_ylabel("Recurrence on the original tumor")
    fig.text(0.01, -0.02, "Dot = patient. Original tumor = all of its segmented parts.", color=FAINT, fontsize=9)
    fig.subplots_adjust(wspace=0.12)
    save(fig, path)


def volume_by_score(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (code, part) in zip(axes, PARTS):
        for shift, key, color, label in ((-0.2, f"first_{code}", SOFT, "Original"), (0.2, code, ORANGE, "Recurrence")):
            groups = [[(r[key]["ml"] if isinstance(r[key], dict) else r[key]) for r in rows
                       if r["score"] == s and r[key] is not None] for s in SCORES]
            ax.boxplot([g or [np.nan] for g in groups], positions=[s + shift for s in SCORES], widths=0.32,
                       showfliers=False, patch_artist=True,
                       boxprops={"facecolor": "white", "edgecolor": color},
                       whiskerprops={"color": color}, capprops={"color": color},
                       medianprops={"color": color, "lw": 2.2})
            rng = np.random.default_rng(0)
            for s, g in zip(SCORES, groups):
                ax.scatter(s + shift + rng.uniform(-0.07, 0.07, len(g)), g, s=9, color=PATIENT, lw=0, zorder=3)
            ax.text(0.62 if shift < 0 else 0.82, 1.03, label, color=color, transform=ax.transAxes,
                    ha="left", fontsize=10)
        ax.set_xticks(SCORES)
        ax.set_xticklabels(SCORES)
        ax.set_xlim(0.4, 6.6)
        ax.set_ylim(bottom=0)
        ax.set_title(part, loc="left", fontsize=11)
        ax.set_xlabel("Pathology score")
    axes[0].set_ylabel("Volume (mL)")
    fig.text(0.01, -0.07, "Box = middle 50 percentile, line = median, whiskers = 1.5 × box height. "
             f"Dot = patient.\n{NOTE}", color=FAINT, fontsize=9)
    fig.subplots_adjust(wspace=0.15)
    save(fig, path)


def margin_by_score(rows, path, absolute):
    fig, axes = plt.subplots(2, len(MARGINS), figsize=(17, 7), sharex=True, sharey="row" if absolute else True)
    for i, (code, part) in enumerate(PARTS):
        for j, mm in enumerate(MARGINS):
            ax = axes[i, j]
            groups = [[r[code]["pct"][mm] * (r[code]["ml"] / 100 if absolute else 1)
                       for r in rows if r["score"] == s and r[code]] for s in SCORES]
            boxes(ax, groups, BLUE)
            ax.set_title(f"{part}, {mm} mm", loc="left", fontsize=11)
            if not absolute:
                ax.set_ylim(0, 102)
                ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        if absolute:
            # the row shares one scale, so it has to reach the largest patient in it
            top = max(r[code]["ml"] for r in rows if r[code])
            axes[i, 0].set_ylim(0, 1.04 * top)
    label = "Recurrence inside the margin (mL)" if absolute else "Recurrence inside the margin (% of volume)"
    fig.supylabel(label, x=0.02, fontsize=10, color=SOFT)
    fig.supxlabel("Pathology score", y=0.06, fontsize=10, color=SOFT)
    fig.text(0.02, 0.005, "Margin = isotropic expansion of the whole original tumor, clipped to the brain. "
             "Box = middle 50 percentile, line = median. Dot = patient.\n" + NOTE, color=FAINT, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.15, wspace=0.12, hspace=0.3)
    save(fig, path)


def main():
    style()
    for space in measure.SPACES:
        rows = collect(space)
        out = results_dir(space, "figures")
        overlap_vs_size(rows, out / "overlap_vs_size.png")
        volume_by_score(rows, out / "volume_by_pathology.png")
        margin_by_score(rows, out / "margin_pct_by_pathology.png", absolute=False)
        margin_by_score(rows, out / "margin_ml_by_pathology.png", absolute=True)


if __name__ == "__main__":
    main()
