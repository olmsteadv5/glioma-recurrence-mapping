#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# How much of the later tumor a uniform margin around the first would catch,
# for every margin from 0 to 70 mm.

import csv
import math
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

# code, name in the files, name on the figure, color. Core follows enhancing
# closely, so the overview draws it thin and dashed
PARTS = (("ET", "enhancing", "Enhancing tumor", "#2a78d6"),
         ("TC", "core", "Tumor core", "#1baf7a"),
         ("WT", "whole", "Whole abnormality", "#eb6834"))
TOP_MM = 70
FINE = np.arange(0, TOP_MM + 0.25, 0.25)
AT_MM = (0, 5, 10, 15, 20, 30, 50, 70)
SHARES = (0.50, 0.90, 0.95, 1.00)

INK, SOFT, FAINT, RULE, PATIENT = "#1f2328", "#5c636b", "#8a9199", "#e3e6e9", "#c3c8cd"


def capture(far, mm):
    return np.searchsorted(far, mm, side="right") / far.size


def margin_for(far, share):
    # the smallest margin that catches this share; inf if the brain edge stops it
    return float(far[math.ceil(share * far.size) - 1])


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


def write_summary(reach, path):
    rows = []
    for code, name, *_ in PARTS:
        fars = list(reach[code].values())
        row = {"part": name, "patients": len(fars)}
        for mm in AT_MM:
            row[f"pct_captured_at_{mm}mm"] = round(100 * float(np.median([capture(f, mm) for f in fars])), 1)
        for share in SHARES:
            need = float(np.median([margin_for(f, share) for f in fars]))
            row[f"margin_mm_for_{int(share * 100)}pct"] = round(need, 1) if np.isfinite(need) else "not reached"
        rows.append(row)
    with open(path, "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    return rows


def style():
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": SOFT, "axes.edgecolor": FAINT,
        "xtick.color": SOFT, "ytick.color": SOFT, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": RULE, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})


def frame(ax):
    ax.set_xlim(0, TOP_MM)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(0, TOP_MM + 1, 10))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.axvline(MARGIN_MM, color=SOFT, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(MARGIN_MM + 1, 0.03, "2 cm plan", color=SOFT, fontsize=8.5)


def curves(fars):
    return np.array([capture(f, FINE) for f in fars])


def per_patient(reach, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, (code, _, label, color) in zip(axes, PARTS):
        grid = curves(reach[code].values())
        for line in grid:
            ax.plot(FINE, line, color=PATIENT, lw=0.8, zorder=2)
        low, mid, high = np.percentile(grid, [25, 50, 75], axis=0)
        ax.fill_between(FINE, low, high, color=color, alpha=0.18, lw=0, zorder=3)
        ax.plot(FINE, mid, color=color, lw=2.2, zorder=4)
        frame(ax)
        ax.set_title(f"{label}  ·  {len(grid)} patients", loc="left", fontsize=11, color=INK, pad=8)
    axes[0].set_ylabel("Later tumor inside the margin")
    fig.supxlabel("Uniform margin around the whole original tumor (mm)", color=SOFT, fontsize=10, y=0.02)
    fig.text(0.005, -0.09, "Each grey line is one patient. The colored line is the median and the "
             "shaded band holds the middle half of patients.\nA few curves stop just short of 100% because "
             "some of the later tumor sits outside the brain edge, where no plan reaches.",
             color=FAINT, fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def overview(reach, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    # medians only: overlapping bands turn to mud, and the spread is in the
    # per-patient figure
    for code, _, label, color in PARTS:
        grid = curves(reach[code].values())
        mid = np.median(grid, axis=0)
        dash = {"ls": (0, (5, 3)), "lw": 1.4} if code == "TC" else {"lw": 2.2}
        ax.plot(FINE, mid, color=color, zorder=4 if code == "TC" else 3,
                label=f"{label} ({len(grid)} patients)", **dash)
    frame(ax)
    ax.set_xlabel("Uniform margin around the whole original tumor (mm)")
    ax.set_ylabel("Later tumor inside the margin")
    ax.set_title("Median patient", loc="left", fontsize=11, color=INK, pad=8)
    ax.legend(loc="lower right", frameon=False, fontsize=9, handlelength=2.4)
    fig.text(0.005, -0.04, "The median across patients at each margin. The spread between patients is "
             "in the per-patient figure.", color=FAINT, fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def main():
    figures = OUTPUTS / "figures"
    figures.mkdir(exist_ok=True)
    reach = collect()
    write_curves(reach, OUTPUTS / "capture_curves.csv")
    rows = write_summary(reach, OUTPUTS / "capture_summary.csv")
    style()
    per_patient(reach, figures / "capture_per_patient.png")
    overview(reach, figures / "capture_median.png")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
