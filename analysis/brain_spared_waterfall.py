#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: one bar per patient, how much less brain beyond the tumor the walk-down irradiates than the 2 cm margin, for slides

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.brain_spared_figure import load
from paths import results_dir

BG, INK, SOFT, BAR, ACCENT = "#000000", "#EDEDEA", "#8C8F94", "#5A5D62", "#4C8DF6"


def draw(cell, patients, phase, out):
    both = [p for p in patients if (p, phase, "iso") in cell and (p, phase, "walk") in cell]
    cut = np.sort(np.array([1 - cell[p, phase, "walk"] / max(cell[p, phase, "iso"], 1e-6) for p in both]) * 100)[::-1]
    median = float(np.median(cut))
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "svg.fonttype": "none",
                         "savefig.dpi": 300, "figure.facecolor": BG, "savefig.facecolor": BG})
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_facecolor(BG)
    # explainer: bars hang down from zero, deeper means more brain spared
    ax.bar(np.arange(len(cut)), -cut, width=0.72, color=BAR, linewidth=0)
    ax.plot([-1, len(cut)], [0, 0], color=INK, lw=1.2)
    ax.plot([-1, len(cut) + 4], [-median, -median], color=ACCENT, lw=1.6)
    ax.text(len(cut) + 10, -median + 1, f"{median:.0f}%", color=ACCENT, fontsize=30, fontweight="bold", va="center")
    ax.text(len(cut) + 10, -median - 10, "less brain, median", color=ACCENT, fontsize=12, va="center")
    ax.text(0, 6, f"{len(cut)} patients", color=SOFT, fontsize=12, va="bottom")
    ax.text(len(cut), 6, f"less brain in {int((cut > 0).sum())} of {len(cut)}", color=SOFT, fontsize=12,
            va="bottom", ha="right")
    for tick in (25, 50, 75, 100):
        ax.plot([-1, len(cut)], [-tick, -tick], color="#1c1d1f", lw=0.8, zorder=0)
        ax.text(-3, -tick, f"−{tick}%", color=SOFT, fontsize=10, va="center", ha="right")
    ax.set_xlim(-12, len(cut) + 70)
    ax.set_ylim(-102, 14)
    ax.axis("off")
    fig.savefig(f"{out}.png", bbox_inches="tight")
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    plt.close(fig)
    return f"{out}.png", median


def main():
    cell, patients = load()
    folder = results_dir("native").parent / "figures"
    for phase in ("46", "60"):
        print(draw(cell, patients, phase, folder / f"brain_spared_waterfall_{phase}"))


if __name__ == "__main__":
    main()
