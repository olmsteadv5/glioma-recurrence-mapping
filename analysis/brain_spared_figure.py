#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: brain irradiated beyond the tumor under the NRG 2 cm margin against the walk-down, per patient, capture left out on purpose

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.combined import TABLES
from analysis.walkdown import RESCORED, WALK_MM
from paths import cohort_of, results_dir

THEMES = {"light": {"bg": "white", "ink": "#1f2328", "soft": "#5c636b", "faint": "#8a9199", "patient": "#c3c8cd",
                    "nrg": "#8a9199", "walk": "#2a78d6"},
          "dark": {"bg": "#000000", "ink": "#EDEDEA", "soft": "#9a9da2", "faint": "#6b6e73", "patient": "#3a3c40",
                   "nrg": "#9a9da2", "walk": "#4C8DF6"}}
PHASES = (("46", "46 Gy, from the whole tumor"), ("60", "60 Gy boost, from the tumor core"))


def load():
    rows = []
    for name in TABLES.values():
        with open(results_dir("native", RESCORED) / name) as fh:
            rows += [r for r in csv.DictReader(fh) if int(r["walk_mm"]) == WALK_MM and r["target"] in ("iso", "walk")]
    cell = {(r["patient"], r["phase"], r["target"]): float(r["beyond_gtv_ml"]) for r in rows}
    patients = sorted({r["patient"] for r in rows})
    return cell, patients


def draw(theme, cell, patients, out):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": t["faint"],
                         "axes.labelcolor": t["soft"], "xtick.color": t["soft"], "ytick.color": t["soft"],
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                         "axes.titlecolor": t["ink"], "axes.titlelocation": "left", "svg.fonttype": "none",
                         "savefig.bbox": "tight", "savefig.dpi": 300, "figure.facecolor": t["bg"],
                         "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"]})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
    for ax, (phase, title) in zip(axes, PHASES):
        both = [p for p in patients if (p, phase, "iso") in cell and (p, phase, "walk") in cell]
        nrg = np.array([cell[p, phase, "iso"] for p in both])
        walk = np.array([cell[p, phase, "walk"] for p in both])
        for a, b in zip(nrg, walk):
            ax.plot([0, 1], [a, b], color=t["patient"], lw=0.5, zorder=1)
        for x, values, colour in ((0, nrg, t["nrg"]), (1, walk, t["walk"])):
            m = np.median(values)
            ax.plot([x - 0.14, x + 0.14], [m, m], color=colour, lw=3, solid_capstyle="round", zorder=3)
            ax.annotate(f"{m:.0f} mL", (x + (0.17 if x else -0.17), m), va="center", ha="left" if x else "right",
                        fontsize=11, color=colour, fontweight="bold")
        ax.plot([0, 1], [np.median(nrg), np.median(walk)], color=t["walk"], lw=1.6, zorder=2)
        # explainer: each patient's own reduction, so a few huge tumors don't set the headline
        cut = np.median(1 - walk / np.maximum(nrg, 1e-6)) * 100
        fewer = int((walk < nrg).sum())
        cohorts = {c: sum(cohort_of(int(q)) == c for q in both) for c in ("UPenn", "TUM", "LUMIERE")}
        ax.set_title(f"{title}\nmedian {cut:.0f}% less brain per patient\nless in {fewer} of {len(both)} patients",
                     fontsize=9.5)
        ax.set_xticks([0, 1], ["NRG 2 cm", "Walk-down"])
        ax.set_xlim(-0.75, 1.6)
        ax.tick_params(axis="x", length=0, labelsize=10)
        for label, colour in zip(ax.get_xticklabels(), (t["nrg"], t["walk"])):
            label.set_color(colour)
    axes[0].set_ylabel("Brain irradiated beyond the tumor (mL)")
    axes[0].set_ylim(0, None)
    fig.text(0, -0.03, f"Grey line = patient ({', '.join(f'{c} {n}' for c, n in cohorts.items())}). Thick tick = median. "
             f"Walk-down at {WALK_MM} mm along HCP-1065 tracts; NRG = {WALK_MM} mm margin. Capture not shown.",
             color=t["faint"], fontsize=7)
    fig.subplots_adjust(wspace=0.25)
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}")
    plt.close(fig)
    return f"{out}.png"


def main():
    cell, patients = load()
    folder = results_dir("native").parent / "figures"
    for theme in THEMES:
        print(draw(theme, cell, patients, folder / f"brain_spared_{theme}"))


if __name__ == "__main__":
    main()
