#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# How much of the later tumor a uniform margin around the first would catch,
# out to the farthest recurrence any plan could reach.

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from geometry import MARGIN_MM
from paths import OUTPUTS

# code, name in the csv, name on the figure, color
PARTS = (("ET", "enhancing", "Enhancing tumor", "#2a78d6"),
         ("TC", "core", "Tumor core", "#1baf7a"),
         ("WT", "whole", "Whole abnormality", "#eb6834"))
# why: Lee et al.'s class boundaries, the share of the recurrence inside the
#      high-dose region: central 95% or more, in-field 80, marginal 20 [lee99]
LEE = (0.95, 0.80, 0.20)

INK, SOFT, FAINT, RULE, PATIENT = "#1f2328", "#5c636b", "#8a9199", "#e3e6e9", "#c3c8cd"


def capture(far, mm):
    return np.searchsorted(far, mm, side="right") / far.size


def collect():
    reach = {code: {} for code, *_ in PARTS}
    for first, later in measure.pairs():
        for code, *_ in PARTS:
            far = measure.reach(first, later, code)
            if far is not None:
                reach[code][first[-8:-3]] = far
    return reach


def far_end(reach):
    # the axis runs to the next 10 mm past the farthest voxel any plan can reach
    last = max(float(far[np.isfinite(far)].max()) for part in reach.values() for far in part.values())
    return int(np.ceil(last / 10) * 10)


def write_curves(reach, top, path):
    with open(path, "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["patient", "part", "margin_mm", "pct_captured"])
        for code, name, *_ in PARTS:
            for patient, far in sorted(reach[code].items()):
                for mm in range(top + 1):
                    out.writerow([patient, name, mm, round(100 * capture(far, mm), 1)])


def style():
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": SOFT, "axes.edgecolor": FAINT,
        "xtick.color": SOFT, "ytick.color": SOFT, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "axes.axisbelow": True,
        "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})


def figure(reach, top, path):
    fine = np.arange(0, top + 0.25, 0.25)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.4), sharex=True)
    for ax, (code, _, label, color) in zip(axes, PARTS):
        grid = np.array([capture(far, fine) for far in reach[code].values()])
        for line in grid:
            ax.plot(fine, line, color=PATIENT, lw=0.8, zorder=2)
        low, mid, high = np.percentile(grid, [25, 50, 75], axis=0)
        ax.fill_between(fine, low, high, color=color, alpha=0.18, lw=0, zorder=3)
        ax.plot(fine, mid, color=color, lw=2.2, zorder=4)
        ax.set_xlim(0, top)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.axvline(MARGIN_MM, color=SOFT, lw=0.9, ls=(0, (4, 3)), zorder=1)
        # the Lee cutoffs stand in for gridlines; the right edge says where the
        # median reaches each one
        # found on a finer grid than the drawing, so the label is not the grid's step
        exact = np.arange(0, top + 0.05, 0.05)
        middle = np.median([capture(far, exact) for far in reach[code].values()], axis=0)
        for share in LEE:
            ax.axhline(share, color=RULE, lw=1.0, zorder=1)
            mm = round(float(exact[np.argmax(middle >= share)]), 1)
            if mm:
                ax.plot(mm, share, "o", ms=4, color=color, mec="white", mew=1, zorder=5)
            ax.text(top * 1.012, share, f"{100 * share:.0f}% at {mm:g} mm", color=SOFT,
                    fontsize=8.5, va="center")
        ax.text(MARGIN_MM + 0.6, 0.05, f"{MARGIN_MM} mm", color=SOFT, fontsize=8.5)
        # the median patient's capture where the median crosses the 20 mm line
        at = float(np.median([capture(far, MARGIN_MM) for far in reach[code].values()]))
        ax.plot(MARGIN_MM, at, "o", ms=5.5, color=color, mec="white", mew=1.2, zorder=5)
        ax.text(MARGIN_MM + 0.8, at - 0.1, f"{100 * at:.1f}%".replace(".0%", "%"), color=INK,
                fontsize=9, zorder=6, bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9})
        ax.set_title(f"{label} ({len(grid)})", loc="left", fontsize=11, color=INK, pad=6)
    axes[0].text(top * 1.012, 1.1, "Median reaches", color=INK, fontsize=8.5, va="center")
    axes[-1].set_xticks(range(0, top + 1, 10))
    axes[-1].set_xlabel("Isotropic expansion (mm)")
    fig.supylabel("Recurrence captured (% of volume)", color=SOFT, fontsize=10, x=0.04)
    fig.text(0.075, 0.005, "Grey line = patient. Colored line = median. "
             "Shaded band = middle 50 percentile.", color=FAINT, fontsize=8.5)
    fig.subplots_adjust(hspace=0.42)
    save(fig, path)


def save(fig, path):
    fig.savefig(path, facecolor="white", transparent=False)
    plt.close(fig)
    # no alpha channel at all, since some viewers paint a checkerboard for one
    Image.open(path).convert("RGB").save(path)


def main():
    figures = OUTPUTS / "figures"
    figures.mkdir(exist_ok=True)
    reach = collect()
    top = far_end(reach)
    write_curves(reach, top, OUTPUTS / "capture_curves.csv")
    style()
    figure(reach, top, figures / "capture_per_patient.png")


if __name__ == "__main__":
    main()
