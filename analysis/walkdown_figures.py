#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Two figures for sending: what the walk-down target costs and catches next to
# the NRG 2 cm volume and a ball of its own size, and whether its shape beats
# that ball patient by patient. Reads walkdown.csv, so run walkdown.py first.

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

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


def paired(by, phase, key, also=()):
    # patients with a value for every target, in one array per target. also
    # names other measures a patient must have, so two calls line up
    common = [p for p in by[(phase, "walk")]
              if all(by[(phase, t)][p][k] is not None for t, *_ in ORDER for k in (key, *also))]
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


def shape(by, path):
    # walk-down minus the ball of its own size, so only the shape differs
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    groups = [(phase, key) for phase, _ in PHASES for key in ("ET_pct", "WT_pct")]
    rng = np.random.default_rng(0)
    for k, (phase, key) in enumerate(groups):
        values = paired(by, phase, key)
        diff = values["walk"] - values["ball"]
        better, worse = int((diff > 0.05).sum()), int((diff < -0.05).sum())
        p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        tone = "#2a78d6" if np.median(diff) > 0 and p < 0.05 else SOFT
        ax.scatter(k + rng.uniform(-0.13, 0.13, len(diff)), diff, s=16, color=PATIENT, lw=0, zorder=2)
        ax.plot([k - 0.24, k + 0.24], [np.median(diff)] * 2, color=tone, lw=3, zorder=3)
        shown = "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
        ax.annotate(f"median {np.median(diff):+.1f} points\nbetter {better}, worse {worse}\n{shown}", (k, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=9.5, color=tone if tone != SOFT else INK)
    ax.axhline(0, color=FAINT, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["46 Gy\nenhancing", "46 Gy\nwhole tumor", "60 Gy\nenhancing", "60 Gy\nwhole tumor"])
    ax.set_ylabel("Walk-down minus same-volume ball\n(percentage points of recurrence caught)")
    ax.set_xlim(-0.5, len(groups) - 0.5)
    fig.text(0.01, 0.005, "Dot = patient. Bar = median. Above zero, following the tracts caught more of "
             "the recurrence than spending the same volume evenly. p = Wilcoxon signed-rank.",
             color=FAINT, fontsize=9)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.8, bottom=0.17)
    save(fig, path)


def tradeoff(by, path):
    # per patient, what moving from the NRG 2 cm volume to the walk-down saves
    # in brain against what it gives up in recurrence caught
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.4))
    for i, (phase, title) in enumerate(PHASES):
        for j, (key, part) in enumerate((("WT_pct", "Whole-tumor recurrence"),
                                         ("ET_pct", "Enhancing recurrence"))):
            ax = axes[i, j]
            saved = paired(by, phase, "beyond_gtv_ml", also=(key,))
            caught = paired(by, phase, key)
            x = saved["iso"] - saved["walk"]
            y = caught["walk"] - caught["iso"]
            # within a point is no real loss: edges of the masks move that much
            lost = y < -1
            ax.axhline(0, color=FAINT, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.scatter(x[~lost], y[~lost], s=26, color="#2a78d6", lw=0, zorder=3)
            ax.scatter(x[lost], y[lost], s=26, color="#eb6834", lw=0, zorder=3)
            ax.plot(np.median(x), np.median(y), "o", ms=11, mfc="none", mec=INK, mew=1.6, zorder=4)
            ax.set_xlim(left=0)
            ax.set_ylim(min(y.min() - 3, -5), 3)
            ax.set_xlabel("Brain spared vs NRG 2 cm (mL)")
            ax.set_ylabel("Change in recurrence caught\n(percentage points)")
            ax.set_title(f"{part} ({len(x)})", loc="left", fontsize=11, color=INK, pad=20)
            ax.text(0, 1.02, f"median patient: {np.median(x):.0f} mL spared, "
                    f"{np.median(y):+.1f} points caught \u00b7 within 1 point: {int((~lost).sum())} of {len(x)}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=SOFT)
            if j == 0:
                ax.text(-0.2, 1.2, title, transform=ax.transAxes, fontsize=12.5,
                        weight="bold", color=INK)
    fig.text(0.01, 0.005, "Dot = patient: blue caught within 1 point of the 2 cm volume, orange caught less. "
             "Open circle = median patient. Both targets trimmed at the brain edge and\nthe midline "
             "(NRG rule). Patient space, BraTS 2025 winning segmentation.", color=FAINT, fontsize=9)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.9, bottom=0.1, hspace=0.62, wspace=0.32)
    save(fig, path)


def main():
    style()
    for space in measure.SPACES:
        by = load(space)
        compare(by, results_dir(space) / "walkdown_compare.png")
        shape(by, results_dir(space) / "walkdown_shape.png")
        tradeoff(by, results_dir(space) / "walkdown_tradeoff.png")


if __name__ == "__main__":
    main()
