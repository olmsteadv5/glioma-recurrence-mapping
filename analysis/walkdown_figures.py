#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# The figure for sending: what the walk-down target costs and catches next to
# the NRG 2 cm volume and a ball of its own size. Reads walkdown.csv, so run
# walkdown.py first.

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from analysis.capture_curves import FAINT, INK, PATIENT, SOFT, save, style
from paths import results_dir

ORDER = (("iso", "NRG 2 cm", "#8f6aa8"), ("walk", "Walk-down", "#2a78d6"),
         ("ball", "Same-volume\nball", "#eb6834"))
PHASES = (("46", "46 Gy volume (from the whole tumor)"), ("60", "60 Gy boost (from the tumor core)"))
MEASURES = (("beyond_gtv_ml", "Brain irradiated beyond the tumor (mL)"),
            ("ET_pct", "Enhancing recurrence caught (%)"),
            ("WT_pct", "Whole-tumor recurrence caught (%)"))


def load(space):
    with open(results_dir(space) / "walkdown.csv") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        value = lambda k: None if r[k] in ("", "None") else float(r[k])
        out.setdefault((r["phase"], r["target"]), {})[r["patient"]] = {
            k: value(k) for k, _ in MEASURES}
    return out


def paired(by, phase, key):
    # patients with a value for every target, in one array per target
    common = [p for p in by[(phase, "walk")]
              if all(by[(phase, t)][p][key] is not None for t, *_ in ORDER)]
    return {t: np.array([by[(phase, t)][p][key] for p in common]) for t, *_ in ORDER}


def compare(by, path):
    # one row per phase: each patient's three targets joined by a line
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for i, (phase, title) in enumerate(PHASES):
        for j, (key, label) in enumerate(MEASURES):
            ax = axes[i, j]
            values = paired(by, phase, key)
            x = np.arange(len(ORDER))
            stack = np.vstack([values[t] for t, *_ in ORDER])
            for column in stack.T:
                ax.plot(x, column, color=PATIENT, lw=0.6, alpha=0.7, zorder=1)
            for k, (t, name, color) in enumerate(ORDER):
                middle = float(np.median(values[t]))
                ax.plot([k - 0.18, k + 0.18], [middle, middle], color=color, lw=3, zorder=3)
                shown = f"{middle:.0f} mL" if key == "beyond_gtv_ml" else f"{middle:.1f}%"
                ax.text(k + 0.22, middle, shown, color=color, fontsize=10, va="center", weight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([name for _, name, _ in ORDER])
            ax.set_xlim(-0.4, len(ORDER) - 0.25)
            if key != "beyond_gtv_ml":
                ax.set_ylim(0, 103)
            else:
                ax.set_ylim(bottom=0)
            ax.set_title(label, loc="left", fontsize=11, color=INK)
            if j == 0:
                ax.text(-0.18, 1.18, title, transform=ax.transAxes, fontsize=12.5,
                        weight="bold", color=INK)
    fig.text(0.01, 0.005, f"Line = one patient (41; 40 have enhancing recurrence), colored bar = median. Patient space, "
             "BraTS 2025 winning segmentation. Walk-down = convex hull around the pre-op tumor and the\n"
             "points 20 mm along every atlas white-matter tract leaving it. Same-volume ball = the "
             "pre-op tumor grown evenly to exactly the walk-down's volume. All three are trimmed at\n"
             "the brain edge and the midline (NRG rule).", color=FAINT, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.12, hspace=0.62, wspace=0.3)
    save(fig, path)


def main():
    style()
    for space in measure.SPACES:
        by = load(space)
        compare(by, results_dir(space, "figures") / "walkdown_compare.png")


if __name__ == "__main__":
    main()
