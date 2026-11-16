#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: figures for the deviation-from-healthy analysis, drawn from the saved voxel table and result csvs without refitting anything

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.deviation import BAND_MM, REACH_MM
from paths import results_dir

INK, SOFT, FAINT, RULE, PATIENT = "#1f2328", "#5c636b", "#8a9199", "#e3e6e9", "#c3c8cd"
OUTCOME = {1: ("becomes enhancing", "#2a78d6"), 2: ("becomes non-enhancing", "#eb6834")}
COMPARTMENT = {1: "Inside pre-op edema", 0: "Normal-appearing tissue"}
MAPS = (("md", "MD (ADC)"), ("tissue_md", "Tissue MD (free water removed)"), ("dti_fa", "FA"),
        ("tissue_fa", "Tissue FA (free water removed)"), ("fw", "Free water"))
# decision: a band only counts for a patient when both groups have this many voxels, fewer gives a noisy median
MIN_VOXELS = 20
# decision: a band's median over patients is drawn only when this many patients have voxels there, fewer patients make spikes
MIN_PATIENTS = 10
# decision: patient-level bootstrap for the figure intervals, same unit the analysis resamples
DRAWS = 2000


def style():
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.edgecolor": FAINT,
                         "axes.labelcolor": SOFT, "xtick.color": SOFT, "ytick.color": SOFT, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 9, "axes.titlecolor": INK, "axes.titlelocation": "left",
                         "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300})


def table():
    keep = ["patient", "compartment", "band", "outcome"] + [f"{m}_z" for m, _ in MAPS]
    with np.load(results_dir("native") / "deviation_voxels_upenn.npz") as d:
        return {k: d[k] for k in keep}


def by_patient(t):
    # explainer: index of each patient's voxels, so later loops slice instead of masking six million rows each time
    order = np.argsort(t["patient"], kind="stable")
    ids, starts = np.unique(t["patient"][order], return_index=True)
    return {int(p): order[s:e] for p, s, e in zip(ids, starts, list(starts[1:]) + [order.size])}


def band_medians(t, rows, name, compartment, outcome):
    # explainer: one median per 4 mm band for one patient, nan where the band has too few voxels
    z, band = t[f"{name}_z"][rows], t["band"][rows]
    pick = (t["compartment"][rows] == compartment) & (t["outcome"][rows] == outcome) & np.isfinite(z)
    out = np.full(REACH_MM // BAND_MM, np.nan)
    for b in range(out.size):
        here = z[pick & (band == b)]
        if here.size >= MIN_VOXELS:
            out[b] = np.median(here)
    return out


def matched_shift(t, rows, name, compartment, outcome, absolute):
    # explainer: recurring minus stable median deviation inside each band, averaged over bands by recurring voxel count, so distance can't make the gap
    z, band = t[f"{name}_z"][rows], t["band"][rows]
    z = np.abs(z) if absolute else z
    here = (t["compartment"][rows] == compartment) & np.isfinite(z)
    gaps, weights = [], []
    for b in range(REACH_MM // BAND_MM):
        rec = z[here & (band == b) & (t["outcome"][rows] == outcome)]
        stable = z[here & (band == b) & (t["outcome"][rows] == 0)]
        if rec.size >= MIN_VOXELS and stable.size >= MIN_VOXELS:
            gaps.append(np.median(rec) - np.median(stable))
            weights.append(rec.size)
    return float(np.average(gaps, weights=weights)) if gaps else np.nan


def boot(values, rng):
    values = values[np.isfinite(values)]
    draws = rng.choice(values, (DRAWS, values.size)).mean(1) if values.size else np.array([np.nan])
    return float(np.mean(values)) if values.size else np.nan, np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def by_distance(t, rows, out):
    names = (("md", "MD (ADC)"), ("dti_fa", "FA"))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    x = (np.arange(REACH_MM // BAND_MM) + 0.5) * BAND_MM
    for r, (name, label) in enumerate(names):
        for c, compartment in enumerate((1, 0)):
            ax = axes[r, c]
            ax.axhline(0, color=FAINT, lw=0.8, ls=(0, (4, 3)))
            stable = np.array([band_medians(t, v, name, compartment, 0) for v in rows.values()])
            n = int(np.isfinite(stable).any(1).sum())
            lo, mid, hi = (np.nanpercentile(stable, q, axis=0) for q in (25, 50, 75))
            ax.fill_between(x, lo, hi, color=RULE, lw=0)
            ax.plot(x, mid, color=FAINT, lw=1.8)
            for outcome, (word, colour) in OUTCOME.items():
                bands = np.array([band_medians(t, v, name, compartment, outcome) for v in rows.values()])
                y = np.where(np.isfinite(bands).sum(0) >= MIN_PATIENTS, np.nanmedian(bands, 0), np.nan)
                ax.plot(x, y, color=colour, lw=2.0, marker="o", ms=3)
            ax.set_title(f"{label}, {COMPARTMENT[compartment].lower()} ({n})")
            ax.set_xlim(0, REACH_MM)
            if r == 1:
                ax.set_xlabel("Distance from pre-op tumor (mm)" if compartment == 0 else "Distance from pre-op core (mm)")
            if c == 0:
                ax.set_ylabel("Deviation from distant healthy tissue (z)")
    for k, (word, colour) in enumerate((("stays stable", FAINT),) + tuple(OUTCOME.values())):
        fig.text(0.12 + 0.17 * k, 0.97, word, color=colour, fontsize=8, fontweight="bold")
    fig.text(0, -0.02, f"Line = median over patients of each patient's band median, drawn where {MIN_PATIENTS}+ patients have voxels. Grey band = middle 50% of patients for "
             "stable voxels.\nDashed = same as healthy tissue of the same class 30+ mm away.", color=FAINT, fontsize=7)
    fig.subplots_adjust(top=0.9, wspace=0.3, hspace=0.35)
    return save(fig, out / "deviation_by_distance")


def shifts(t, rows, out):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), sharex=True, sharey=True)
    ticks, labels = [], []
    for i, (name, label) in enumerate(MAPS):
        for j, word in enumerate(("higher or lower", "how far either way")):
            ticks.append(i * 3 + j)
            labels.append(f"{label}, {word}" if j == 0 else word)
    for r, compartment in enumerate((1, 0)):
        for c, (outcome, (word, colour)) in enumerate(OUTCOME.items()):
            ax = axes[r, c]
            ax.axvline(0, color=FAINT, lw=0.8, ls=(0, (4, 3)))
            counts = []
            for i, (name, _) in enumerate(MAPS):
                for j, absolute in enumerate((False, True)):
                    y = i * 3 + j
                    values = np.array([matched_shift(t, v, name, compartment, outcome, absolute) for v in rows.values()])
                    counts.append(int(np.isfinite(values).sum()))
                    jitter = rng.uniform(-0.25, 0.25, values.size)
                    ax.scatter(values, y + jitter, s=5, color=PATIENT, lw=0, zorder=1)
                    centre, lo, hi = boot(values, rng)
                    ax.plot([lo, hi], [y, y], color=colour, lw=2.0, zorder=2, solid_capstyle="round")
                    ax.plot(centre, y, "o", color=colour, ms=5, mfc="white" if absolute else colour, mew=1.5, zorder=3)
            ax.set_title(f"{COMPARTMENT[compartment]}, {word} ({max(counts)})")
            ax.set_yticks(ticks, labels)
            ax.set_xlim(-2, 2)
            if r == 1:
                ax.set_xlabel("Recurring minus stable deviation, same distance (z)")
    # explainer: the axes share y, so one flip turns all four
    axes[0, 0].invert_yaxis()
    fig.text(0, -0.02, "Grey dot = patient. Coloured dot and bar = mean over patients and 95% patient bootstrap.\nFilled = "
             "signed deviation, hollow = size of deviation either way. 0 = recurring tissue looks like stable tissue.",
             color=FAINT, fontsize=7)
    fig.subplots_adjust(wspace=0.12, hspace=0.3)
    return save(fig, out / "deviation_shift")


def aucs(out):
    with open(results_dir("native") / "deviation_auc_upenn.csv") as fh:
        rows = list(csv.DictReader(fh))
    kinds = (("z", "higher than healthy"), ("|z|", "deviates either way"), ("ratio to mirror", "vs mirrored side"))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharex=True, sharey=True)
    ticks, labels = [], []
    for i, (name, label) in enumerate(MAPS):
        for j, (_, word) in enumerate(kinds):
            ticks.append(i * 4 + j)
            labels.append(f"{label}, {word}" if j == 0 else word)
    for r, (compartment, ctitle) in enumerate((("edema", COMPARTMENT[1]), ("normal-appearing", COMPARTMENT[0]))):
        for c, (outcome, (word, colour)) in enumerate(OUTCOME.items()):
            ax = axes[r, c]
            ax.axvline(0.5, color=FAINT, lw=0.8, ls=(0, (4, 3)))
            wanted = word.replace("becomes ", "")
            n = 0
            for i, (name, _) in enumerate(MAPS):
                for j, (kind, _) in enumerate(kinds):
                    hit = [x for x in rows if x["outcome"] == wanted and x["compartment"] == compartment
                           and x["score"] == f"{name} {kind}"]
                    if not hit:
                        continue
                    h = hit[0]
                    n = max(n, int(h["patients"]))
                    y = i * 4 + j
                    ax.plot([float(h["ci_low"]), float(h["ci_high"])], [y, y], color=colour, lw=1.8, solid_capstyle="round")
                    ax.plot(float(h["auc"]), y, "o", color=colour, ms=4.5, mfc=("white", colour, PATIENT)[j], mew=1.3)
            ax.set_title(f"{ctitle}, {word} ({n})")
            ax.set_yticks(ticks, labels)
            ax.set_xlim(0.35, 0.7)
            if r == 1:
                ax.set_xlabel("Within-patient AUC at matched distance")
    axes[0, 0].invert_yaxis()
    fig.text(0, -0.02, "Hollow = higher than healthy, filled = deviates either way, grey fill = ratio to the mirrored point. "
             "Dot = AUC pooled over patients and 4 mm bands, bar = 95% patient bootstrap.\n0.5 = no better than "
             "chance at telling recurring from stable tissue the same distance out.", color=FAINT, fontsize=7)
    fig.subplots_adjust(wspace=0.12, hspace=0.3)
    return save(fig, out / "deviation_auc")


def models(out):
    with open(results_dir("native") / "deviation_model_upenn.csv") as fh:
        rows = list(csv.DictReader(fh))
    steps = ("distance only", "distance + diffusion", "distance + diffusion + tissue md")
    names = ("Distance from tumor", "+ diffusion deviation", "+ tissue MD")
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    for k, (outcome, (word, colour)) in enumerate(OUTCOME.items()):
        wanted = word.replace("becomes ", "")
        values = [float(next(r for r in rows if r["outcome"] == wanted and r["model"] == s)["held_out_auc"]) for s in steps]
        ys = np.arange(3) + k * 4
        ax.plot(values, ys, color=colour, lw=1, alpha=0.5)
        ax.scatter(values, ys, color=colour, s=22, zorder=3)
        for v, y in zip(values, ys):
            ax.annotate(f"{v:.3f}", (v, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=SOFT)
        ax.text(0.52, k * 4 - 0.9, word.capitalize(), color=colour, fontsize=8)
    ax.set_yticks(list(np.arange(3)) + list(np.arange(3) + 4), list(names) * 2)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Held-out AUC, leave one patient out")
    return save(fig, out / "deviation_models")


def save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}")
    plt.close(fig)
    return f"{stem}.png"


def main():
    style()
    out = results_dir("native").parent / "figures"
    print(aucs(out))
    print(models(out))
    t = table()
    rows = by_patient(t)
    print(by_distance(t, rows, out))
    print(shifts(t, rows, out))


if __name__ == "__main__":
    main()
