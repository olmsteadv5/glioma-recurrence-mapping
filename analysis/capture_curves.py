#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# How much of the later tumor a uniform margin around the first would catch,
# for every margin from 0 to 70 mm.

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from geometry import MARGIN_MM
from paths import OUTPUTS

# code, name in the csv, name on the figure, color
PARTS = (("ET", "enhancing", "Enhancing tumor", "#2a78d6"),
         ("TC", "core", "Tumor core", "#1baf7a"),
         ("WT", "whole", "Whole abnormality", "#eb6834"))
TOP_MM = 70
# why: Lee et al.'s classes by the share of the recurrence inside the high-dose
#      region: central 95% or more, in-field 80-95, marginal 20-80, outside [lee99]
LEE = ((0.95, 1.0, "central  ≥95%"), (0.80, 0.95, "in-field  80–95%"),
       (0.20, 0.80, "marginal  20–80%"), (0.0, 0.20, "outside  <20%"))
FINE = np.arange(0, TOP_MM + 0.25, 0.25)

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


def write_curves(reach, path):
    with open(path, "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["patient", "part", "margin_mm", "pct_captured"])
        for code, name, *_ in PARTS:
            for patient, far in sorted(reach[code].items()):
                for mm in range(TOP_MM + 1):
                    out.writerow([patient, name, mm, round(100 * capture(far, mm), 1)])


def style():
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": SOFT, "axes.edgecolor": FAINT,
        "xtick.color": SOFT, "ytick.color": SOFT, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "axes.axisbelow": True,
        "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})


def figure(reach, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.4), sharex=True)
    for ax, (code, _, label, color) in zip(axes, PARTS):
        grid = np.array([capture(far, FINE) for far in reach[code].values()])
        for line in grid:
            ax.plot(FINE, line, color=PATIENT, lw=0.8, zorder=2)
        low, mid, high = np.percentile(grid, [25, 50, 75], axis=0)
        ax.fill_between(FINE, low, high, color=color, alpha=0.18, lw=0, zorder=3)
        ax.plot(FINE, mid, color=color, lw=2.2, zorder=4)
        ax.set_xlim(0, TOP_MM)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.axvline(MARGIN_MM, color=SOFT, lw=0.9, ls=(0, (4, 3)), zorder=1)
        # the Lee cutoffs stand in for gridlines, named at the right edge
        for low_share, high_share, name in LEE:
            if low_share:
                ax.axhline(low_share, color=RULE, lw=1.0, zorder=1)
            ax.text(TOP_MM + 1, (low_share + high_share) / 2, name, color=SOFT,
                    fontsize=8.5, va="center")
        ax.text(MARGIN_MM + 0.6, 0.05, f"{MARGIN_MM} mm", color=SOFT, fontsize=8.5)
        # the median patient's capture where the median crosses the 20 mm line
        at = float(np.median([capture(far, MARGIN_MM) for far in reach[code].values()]))
        ax.plot(MARGIN_MM, at, "o", ms=5.5, color=color, mec="white", mew=1.2, zorder=5)
        ax.text(MARGIN_MM + 0.8, at - 0.1, f"{100 * at:.1f}%".replace(".0%", "%"), color=INK,
                fontsize=9, zorder=6, bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9})
        ax.set_title(f"{label} ({len(grid)})", loc="left", fontsize=11, color=INK, pad=6)
    axes[-1].set_xticks(range(0, TOP_MM + 1, 10))
    axes[-1].set_xlabel("Isotropic expansion (mm)")
    fig.supylabel("Recurrence captured (% of volume)", color=SOFT, fontsize=10, x=0.04)
    fig.text(0.075, 0.005, "Grey line = patient. Colored line = median. "
             "Shaded band = middle 50 percentile.", color=FAINT, fontsize=8.5)
    fig.subplots_adjust(hspace=0.42)
    fig.savefig(path)
    plt.close(fig)


def main():
    figures = OUTPUTS / "figures"
    figures.mkdir(exist_ok=True)
    reach = collect()
    write_curves(reach, OUTPUTS / "capture_curves.csv")
    style()
    figure(reach, figures / "capture_per_patient.png")


if __name__ == "__main__":
    main()
