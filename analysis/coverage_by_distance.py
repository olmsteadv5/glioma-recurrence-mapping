#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: share of brain the diffusion scan reached in 1 mm shells around each pre-op tumor part, one line per patient

import csv
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import out_dir, results_dir
from scan.diffusion import PREP

INK, SOFT, FAINT, RULE, PATIENT = "#1f2328", "#5c636b", "#8a9199", "#e3e6e9", "#c3c8cd"
PARTS = (("enhancing", (3,), "Enhancing tumor", "#2a78d6"),
         ("non-enhancing", (1, 2), "Non-enhancing tumor and edema", "#eb6834"))
REACH_MM = 30
# decision: a voxel counts as covered at the same cut the maps use to blank it
SEEN = 0.5
# decision: patients whose worst shell drops below this get their number on the chart
LABEL_BELOW = 0.9


def shells(subject):
    d = out_dir(subject)
    image = nib.load(d / "seg.nii.gz")
    seg = np.asarray(image.dataobj)
    brain = np.asarray(nib.load(d / "brainmask.nii.gz").dataobj) > 0
    seen = np.nan_to_num(np.asarray(nib.load(d / "diffusion_prep" / "coverage_on_t2.nii.gz").dataobj)) >= SEEN
    zooms = image.header.get_zooms()[:3]
    out = {}
    for name, labels, *_ in PARTS:
        part = np.isin(seg, labels)
        if not part.any():
            continue
        far = ndimage.distance_transform_edt(~part, sampling=zooms)
        # explainer: shell 0 is the tumor part itself, shell k is everything more than k-1 and up to k mm out
        band = np.ceil(far).astype(np.int32)
        keep = brain & (band <= REACH_MM)
        total = np.bincount(band[keep], minlength=REACH_MM + 1)
        hit = np.bincount(band[keep & seen], minlength=REACH_MM + 1)
        out[name] = np.where(total > 0, hit / np.maximum(total, 1), np.nan)
    return out


def plot(curves, path):
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.edgecolor": FAINT,
                         "axes.labelcolor": SOFT, "xtick.color": SOFT, "ytick.color": SOFT, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 9, "axes.titlecolor": INK,
                         "axes.titlelocation": "left", "svg.fonttype": "none", "savefig.bbox": "tight", "savefig.dpi": 300})
    x = np.arange(REACH_MM + 1)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, sharey=True)
    for ax, (name, _, title, colour) in zip(axes, PARTS):
        rows = {s: c[name] for s, c in curves.items() if name in c}
        Y = np.array(list(rows.values()))
        for y in Y:
            ax.plot(x, y * 100, color=PATIENT, lw=0.7, zorder=1)
        lo, mid, hi = (np.nanpercentile(Y, q, axis=0) * 100 for q in (25, 50, 75))
        ax.fill_between(x, lo, hi, color=colour, alpha=0.15, lw=0, zorder=2)
        ax.plot(x, mid, color=colour, lw=2.2, zorder=3)
        ax.annotate("median", (x[-1], mid[-1]), xytext=(34, 0), textcoords="offset points", color=colour, fontsize=7,
                    va="center", annotation_clip=False)
        # explainer: name every patient that drops below the line, at its low point if that's early, else stacked at the right edge
        short = lambda s: s.replace("UPENN-GBM-", "").replace("_11", "")
        low = {s: y for s, y in rows.items() if np.nanmin(y) < LABEL_BELOW}
        edge = []
        for s, y in low.items():
            at = int(np.nanargmin(y))
            if at < REACH_MM - 5:
                ax.plot(at, y[at] * 100, "o", color=INK, ms=2.5, zorder=4)
                ax.annotate(short(s), (at, y[at] * 100), xytext=(4, -2), textcoords="offset points", color=INK,
                            fontsize=6.5, va="top")
            else:
                ax.plot(x[-1], y[-1] * 100, "o", color=INK, ms=2.5, zorder=4)
                edge.append((y[-1] * 100, s))
        # explainer: labels at the edge are pushed apart so none sit on top of each other
        placed = []
        for value, s in sorted(edge, reverse=True):
            spot = min(value, placed[-1] - 2.4) if placed else value
            placed.append(spot)
            ax.annotate(short(s), (x[-1], value), xytext=(x[-1] + 1.6, spot), textcoords="data", color=INK, fontsize=6.5,
                        va="center", annotation_clip=False,
                        arrowprops={"arrowstyle": "-", "color": FAINT, "lw": 0.5, "shrinkA": 0, "shrinkB": 2})
        ax.set_title(f"{title} ({len(rows)} pre-op scans)")
        for g in (90, 100):
            ax.axhline(g, color=RULE, lw=0.8, zorder=0)
        ax.set_ylim(min(60, np.nanmin(Y) * 100 - 3), 101.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    axes[-1].set_xlim(0, REACH_MM)
    axes[-1].set_xlabel("Distance from the tumor part (mm), 0 = inside it")
    fig.supylabel("Brain in the shell the diffusion scan reached", color=SOFT, fontsize=8)
    fig.text(0, -0.02, "Grey line = patient. Coloured line = median, band = middle 50%. Coverage = voxels with coverage "
             f">= {SEEN}, the cut the maps use.\nShells are 1 mm wide and only count brain. Numbered dots = the lowest "
             f"point of patients that drop below {int(LABEL_BELOW * 100)}%.", color=FAINT, fontsize=7)
    fig.subplots_adjust(hspace=0.3, right=0.86)
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}")
    return f"{path}.png"


def main():
    scans = sorted(p.name for p in PREP.iterdir()
                   if p.name.endswith("_11") and (out_dir(p.name) / "diffusion_prep" / "coverage_on_t2.nii.gz").exists())
    curves = {s: shells(s) for s in scans}
    table = results_dir("native") / "coverage_by_distance.csv"
    with open(table, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scan", "part"] + [f"{k}mm" for k in range(REACH_MM + 1)])
        for s, c in curves.items():
            for name, y in c.items():
                writer.writerow([s, name] + [round(float(v), 4) for v in y])
    print(table)
    print(plot(curves, results_dir("native").parent / "figures" / "coverage_by_distance"))


if __name__ == "__main__":
    main()
