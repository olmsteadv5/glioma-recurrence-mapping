#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: walk-down along each patient's own diffusion tracts against walk-down along HCP-1065 atlas tracts, same patients and same scoring

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.walkdown import RESCORED, WALK_MM
from paths import results_dir

INK, SOFT, FAINT, RULE, PATIENT = "#1f2328", "#5c636b", "#8a9199", "#e3e6e9", "#c3c8cd"
# decision: both scored only where the diffusion scan reached, so the comparison is the tracts and nothing else
SOURCES = (("walkdown_dtiprep_covered_curves.csv", "Patient diffusion tracts", "#eb6834"),
           ("walkdown_covered_curves.csv", "HCP-1065 atlas tracts", "#2a78d6"))
PHASE = "46"
num = lambda v: None if v in ("", "None") else float(v)


def load(name):
    with open(results_dir("native", RESCORED) / name) as fh:
        rows = [r for r in csv.DictReader(fh) if r["phase"] == PHASE]
    return {(r["patient"], int(r["walk_mm"]), r["target"]): r for r in rows}


def gain(cell, patient, mm, column):
    walk, ball = cell.get((patient, mm, "walk")), cell.get((patient, mm, "ball"))
    if walk is None or ball is None or num(walk[column]) is None or num(ball[column]) is None:
        return np.nan
    return num(walk[column]) - num(ball[column])


def style():
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.edgecolor": FAINT,
                         "axes.labelcolor": SOFT, "xtick.color": SOFT, "ytick.color": SOFT, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 9, "axes.titlecolor": INK,
                         "axes.titlelocation": "left", "svg.fonttype": "none", "savefig.bbox": "tight", "savefig.dpi": 300})


def slope(ax, left, right, title, ylabel, zero=False):
    # explainer: one grey line per patient joining its two values, the coloured dots and bar are the medians
    for a, b in zip(left, right):
        if np.isfinite(a) and np.isfinite(b):
            ax.plot([0, 1], [a, b], color=PATIENT, lw=0.7, zorder=1)
    for x, values, (_, label, colour) in ((0, left, SOURCES[0]), (1, right, SOURCES[1])):
        ax.scatter(np.full(len(values), x), values, s=7, color=colour, alpha=0.6, lw=0, zorder=2)
        m = np.nanmedian(values)
        ax.plot([x - 0.12, x + 0.12], [m, m], color=colour, lw=2.4, zorder=3, solid_capstyle="round")
        ax.annotate(f"{m:.1f}", (x + 0.14, m), va="center", fontsize=7, color=colour)
    if zero:
        ax.axhline(0, color=FAINT, lw=0.8, ls=(0, (4, 3)), zorder=0)
    both = np.isfinite(left) & np.isfinite(right)
    d = np.asarray(right)[both] - np.asarray(left)[both]
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    ax.set_title(f"{title}\natlas minus patient\nmedian {np.median(d):+.1f}, p = {p:.2g}", fontsize=8)
    ax.set_xticks([0, 1], ["Patient\ntracts", "Atlas\ntracts"])
    ax.set_xlim(-0.35, 1.45)
    ax.set_ylabel(ylabel)


def main():
    style()
    cells = [load(name) for name, *_ in SOURCES]
    patients = sorted({k[0] for k in cells[0]} & {k[0] for k in cells[1]})
    fig = plt.figure(figsize=(7.2, 6.6))
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 0.95), hspace=0.55, wspace=0.55)
    vol = [[num(c[(p, WALK_MM, "walk")]["volume_ml"]) for p in patients] for c in cells]
    slope(fig.add_subplot(grid[0, 0]), np.array(vol[0]), np.array(vol[1]), "Target volume", "46 Gy target (mL)")
    for k, (column, title) in enumerate((("WT_pct", "FLAIR recurrence caught"), ("ET_pct", "Enhancing caught"))):
        values = [np.array([gain(c, p, WALK_MM, column) for p in patients]) for c in cells]
        slope(fig.add_subplot(grid[0, k + 1]), values[0], values[1], f"{title}, vs ball",
              "Walk-down minus ball (points)", zero=True)
    ax = fig.add_subplot(grid[1, :])
    distances = sorted({k[1] for k in cells[0]} & {k[1] for k in cells[1]})
    for cell, (_, label, colour) in zip(cells, SOURCES):
        Y = np.array([[gain(cell, p, mm, "WT_pct") for mm in distances] for p in patients])
        lo, mid, hi = (np.nanpercentile(Y, q, axis=0) for q in (25, 50, 75))
        ax.fill_between(distances, lo, hi, color=colour, alpha=0.13, lw=0)
        ax.plot(distances, mid, color=colour, lw=2.2)
        # explainer: label each line where it sits furthest from the other, the atlas above at its peak, the patient line below
        at = int(np.nanargmax(mid)) if colour == SOURCES[1][2] else distances.index(WALK_MM + 6)
        ax.annotate(label, (distances[at], mid[at]), xytext=(0, 8 if colour == SOURCES[1][2] else -10),
                    textcoords="offset points", color=colour, fontsize=7.5, ha="center",
                    va="bottom" if colour == SOURCES[1][2] else "top")
    ax.axhline(0, color=FAINT, lw=0.8, ls=(0, (4, 3)))
    ax.axvline(WALK_MM, color=FAINT, lw=0.8)
    ax.annotate(f"{WALK_MM} mm, used above", (WALK_MM, ax.get_ylim()[1]), xytext=(4, -2), textcoords="offset points",
                color=FAINT, fontsize=7, va="top")
    ax.set_xlim(distances[0], distances[-1])
    ax.set_xlabel("Walk distance along the tracts (mm)")
    ax.set_ylabel("FLAIR recurrence, walk-down\nminus ball (points)")
    ax.set_title("Gain over the same-volume ball at every walk distance")
    fig.text(0, 0.0, f"{len(patients)} UPenn patients with pre-op diffusion, 46 Gy target, both scored only where the "
             "diffusion scan reached. Grey line = patient.\nThick tick = median. Bottom: line = median over patients, "
             "band = middle 50%. p = paired Wilcoxon, atlas against patient tracts.", color=FAINT, fontsize=7,
             va="top", transform=fig.transFigure)
    fig.subplots_adjust(right=0.95, bottom=0.1)
    out = results_dir("native").parent / "figures" / "walkdown_sources"
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}")
    print(f"{out}.png")


if __name__ == "__main__":
    main()
