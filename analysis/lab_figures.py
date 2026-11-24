#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: the lab-meeting chart set, one style and one claim per chart, numbered in the order they are presented

import csv
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from analysis.walkdown import RESCORED, WALK_MM
from geometry import MARGIN_MM
from paths import cohort_of, results_dir

ZONES = {"dark": ("#3A5A8C", "#4C8DF6", "#F08A50", "#E5484D"), "light": ("#A9C4EC", "#2A78D6", "#E0682C", "#C93A3F")}
THEMES = {"dark": {"bg": "#000000", "ink": "#EDEDEA", "soft": "#A3A6AB", "faint": "#6B6E73", "rule": "#26282B",
                   "patient": "#34363A", "walk": "#4C8DF6", "other": "#F08A50", "base": "#A3A6AB"},
          "light": {"bg": "#FFFFFF", "ink": "#1F2328", "soft": "#5C636B", "faint": "#8A9199", "rule": "#E3E6E9",
                    "patient": "#D3D7DB", "walk": "#2A78D6", "other": "#E0682C", "base": "#8A9199"}}
T = THEMES["dark"]
PHASES = {"46": "46 Gy", "60": "60 Gy boost"}
KINDS = {"WT_pct": "FLAIR recurrence", "ET_pct": "Enhancing recurrence"}
COHORTS = ("UPenn", "TUM", "LUMIERE")
RNG = np.random.default_rng(0)
num = lambda v: None if v in ("", "None", None) else float(v)


# ---------- data ----------

def curves(*names):
    rows = []
    for name in names:
        with open(results_dir("native", RESCORED) / name) as fh:
            rows += list(csv.DictReader(fh))
    return {(r["patient"], r["phase"], int(r["walk_mm"]), r["target"]): r for r in rows}


def patients_of(cell):
    return sorted({k[0] for k in cell})


def value(cell, p, phase, mm, target, column):
    row = cell.get((p, phase, mm, target))
    return num(row[column]) if row else None


def gains(cell, patients, phase, column, mm=WALK_MM):
    out = []
    for p in patients:
        a, b = value(cell, p, phase, mm, "walk", column), value(cell, p, phase, mm, "ball", column)
        if a is not None and b is not None:
            out.append(a - b)
    return np.array(out)


def mean_ci(values, draws=4000):
    if not len(values):
        return np.nan, np.nan, np.nan
    boot = RNG.choice(values, (draws, len(values))).mean(1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def p_of(values):
    return float(wilcoxon(values).pvalue) if np.any(values != 0) else 1.0


def pstr(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"


# ---------- style ----------

def style():
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 10,
                         "axes.edgecolor": T["faint"], "axes.labelcolor": T["soft"], "xtick.color": T["soft"],
                         "ytick.color": T["soft"], "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 11, "axes.titlecolor": T["ink"], "axes.titlelocation": "left",
                         "svg.fonttype": "none", "savefig.dpi": 250, "figure.facecolor": T["bg"],
                         "axes.facecolor": T["bg"], "savefig.facecolor": T["bg"], "legend.frameon": False})


def headline(fig, claim, sub):
    fig.text(0.0, 1.0, claim, fontsize=15, fontweight="bold", color=T["ink"], va="bottom", transform=fig.transFigure)
    fig.text(0.0, 0.985, sub, fontsize=10, color=T["soft"], va="top", transform=fig.transFigure)


def note(fig, text):
    fig.text(0.0, -0.02, text, fontsize=8, color=T["faint"], va="top", transform=fig.transFigure)


def save(fig, folder, name):
    for ext in ("png", "svg"):
        fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(folder / f"{name}.png", flush=True)


# ---------- part 1: the atlas walk-down, old results restated ----------

def f01_reach(folder):
    # explainer: share of each later recurrence a uniform margin of each size would catch, from the current segmentations
    reach = {"ET": [], "WT": []}
    for first, later in measure.pairs("native"):
        for code in reach:
            far = measure.reach(first, later, code, "native")
            if far is not None and len(far):
                reach[code].append(np.sort(far))
        measure.forget()
    mm = np.arange(0, 31)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (code, title) in zip(axes, (("ET", "Enhancing recurrence"), ("WT", "FLAIR recurrence"))):
        Y = np.array([np.searchsorted(f, mm, side="right") / len(f) * 100 for f in reach[code]])
        for y in Y:
            ax.plot(mm, y, color=T["patient"], lw=0.7)
        med = np.median(Y, 0)
        ax.plot(mm, med, color=T["walk"], lw=2.6)
        at = mm[np.argmax(med >= 95)] if (med >= 95).any() else None
        ax.axvline(20, color=T["faint"], lw=1, ls=(0, (4, 3)))
        ax.text(20.4, 4, "NRG 2 cm", color=T["faint"], fontsize=9)
        if at is not None:
            ax.plot(at, 95, "o", color=T["walk"], ms=7)
            ax.annotate(f"95% within {at} mm", (at, 95), xytext=(8, -18), textcoords="offset points",
                        color=T["walk"], fontsize=11, fontweight="bold")
        ax.set_title(f"{title} ({len(Y)} patients)")
        ax.set_xlabel("Uniform margin around the pre-op tumor (mm)")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("Recurrence caught (%)")
    headline(fig, "Enhancing recurrence stays close; FLAIR recurrence spreads further",
             "How much of each later recurrence a uniform margin catches, UPenn")
    note(fig, "Grey line = patient. Blue = median.")
    save(fig, folder, "01_where_recurrence_goes")


def f02_location(folder, theme):
    # explainer: every follow-up tumor split by where it sits relative to the original whole tumor
    rows = []
    for first, later in measure.pairs("native"):
        rows.append({code: measure.zones(first, later, code, "native") for code in ("ET", "WT")})
        measure.forget()
    names = (("inside", "Inside the original tumor"), ("within", f"Outside, within {MARGIN_MM} mm"),
             ("beyond", f"Beyond {MARGIN_MM} mm, connected"), ("separate", f"Separate lesion beyond {MARGIN_MM} mm"))
    colours = dict(zip([k for k, _ in names], ZONES[theme]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.4))
    summary = {}
    for ax, (code, title) in zip(axes, (("ET", "Enhancing recurrence"), ("WT", "FLAIR recurrence"))):
        shares = []
        for r in rows:
            z = r[code]
            total = sum(z.values()) if z else 0
            if total > 0:
                shares.append({k: z[k] / total * 100 for k, _ in names})
        # decision: sorted by how much lies outside the original tumor, so the spread reads as one sweep
        shares.sort(key=lambda d: (d["beyond"] + d["separate"], d["within"]))
        for i, d in enumerate(shares):
            left = 0
            for k, _ in names:
                ax.barh(i, d[k], left=left, height=0.78, color=colours[k], linewidth=0)
                left += d[k]
        far = np.array([d["beyond"] + d["separate"] for d in shares])
        summary[code] = (np.median([d["inside"] for d in shares]), np.median(far), int((far > 10).sum()), len(shares))
        ax.set_title(f"{title} ({len(shares)} patients)\n{int((far > 10).sum())} with over 10% beyond {MARGIN_MM} mm",
                     fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, len(shares) - 0.4)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Share of the follow-up tumor (%)")
    handles = [matplotlib.patches.Patch(color=colours[k], label=label) for k, label in names]
    # explainer: key across the top, under the headline, so it never meets the footnote
    fig.legend(handles=handles, loc="upper left", ncol=4, bbox_to_anchor=(0.0, 0.955), labelcolor=T["soft"],
               fontsize=9, handlelength=1.2)
    fig.subplots_adjust(top=0.84)
    et, wt = summary["ET"], summary["WT"]
    headline(fig, "Recurrence grows back in or right next to the original tumor",
             f"Median share inside the original tumor: {et[0]:.0f}% of enhancing and {wt[0]:.0f}% of FLAIR recurrence. "
             f"Median beyond {MARGIN_MM} mm: {et[1]:.0f}% and {wt[1]:.0f}%")
    note(fig, "Bar = one UPenn patient, sorted by the share beyond 20 mm. Distance is from the edge of the original whole "
              "tumor (FLAIR + core). Separate lesion = a piece of recurrence not connected to anything within 20 mm.")
    save(fig, folder, "02_recurrence_location")


def f02_brain(folder, cell):
    patients = patients_of(cell)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, phase in zip(axes, PHASES):
        both = [p for p in patients if value(cell, p, phase, WALK_MM, "iso", "beyond_gtv_ml") is not None
                and value(cell, p, phase, WALK_MM, "walk", "beyond_gtv_ml") is not None]
        nrg = np.array([value(cell, p, phase, WALK_MM, "iso", "beyond_gtv_ml") for p in both])
        walk = np.array([value(cell, p, phase, WALK_MM, "walk", "beyond_gtv_ml") for p in both])
        for a, b in zip(nrg, walk):
            ax.plot([0, 1], [a, b], color=T["patient"], lw=0.5)
        for x, v, c in ((0, nrg, T["base"]), (1, walk, T["walk"])):
            m = np.median(v)
            ax.plot([x - 0.14, x + 0.14], [m, m], color=c, lw=3.5, solid_capstyle="round")
            ax.text(x + (0.2 if x else -0.2), m, f"{m:.0f} mL", color=c, fontsize=12, fontweight="bold",
                    va="center", ha="left" if x else "right")
        cut = np.median(1 - walk / np.maximum(nrg, 1e-6)) * 100
        ax.set_title(f"{PHASES[phase]}: {cut:.0f}% less per patient, less in {int((walk < nrg).sum())} of {len(both)}")
        ax.set_xticks([0, 1], ["NRG 2 cm", "Walk-down"])
        ax.set_xlim(-0.8, 1.7)
    axes[0].set_ylabel("Brain irradiated beyond the tumor (mL)")
    headline(fig, "The walk-down irradiates less brain than the 2 cm margin",
             f"{len(patients)} patients: UPenn, TUM, LUMIERE")
    note(fig, "Grey line = patient. Thick tick = median.")
    save(fig, folder, "03_brain_irradiated")


def f03_equal_volume(folder, cell):
    # explainer: medians over patients at each grow distance, so each line traces brain cost against recurrence caught
    patients = patients_of(cell)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    lines = (("iso", "Uniform margin", T["base"]), ("ball", "Same-volume ball", T["other"]), ("walk", "Walk-down", T["walk"]))
    for ax, (column, title) in zip(axes, KINDS.items()):
        for target, label, colour in lines:
            xs, ys = [], []
            for mm in range(0, 41):
                pts = [(value(cell, p, "46", mm, target, "beyond_gtv_ml"), value(cell, p, "46", mm, target, column))
                       for p in patients]
                pts = [(a, b) for a, b in pts if a is not None and b is not None]
                if len(pts) > len(patients) // 2:
                    xs.append(np.median([a for a, _ in pts]))
                    ys.append(np.median([b for _, b in pts]))
            ax.plot(xs, ys, color=colour, lw=2.4 if target == "walk" else 1.8, label=label)
        ax.set_title(f"{title}, 46 Gy")
        ax.set_xlabel("Brain irradiated beyond the tumor, median (mL)")
        ax.set_ylim(40 if column == "WT_pct" else 70, 101)
    axes[0].set_ylabel("Recurrence caught, median (%)")
    axes[0].legend(loc="lower right", labelcolor=T["soft"], fontsize=9)
    headline(fig, "Any bigger target catches more, so compare shapes at the same volume",
             "Each line grows one way: the walk-down, a ball of exactly its volume, and a uniform margin")
    note(fig, "Points along each line are grow distances 0 to 40 mm. Where the blue line sits above the orange, "
              "the tract shape caught more for the same brain.")
    save(fig, folder, "04_compare_at_equal_volume")


def f04_by_cohort(folder, cell):
    patients = patients_of(cell)
    groups = {c: [p for p in patients if cohort_of(int(p)) == c] for c in COHORTS}
    groups["All 3 pooled"] = patients
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex="col")
    for r, phase in enumerate(PHASES):
        for c, (column, title) in enumerate(KINDS.items()):
            ax = axes[r, c]
            ax.axvline(0, color=T["faint"], lw=1, ls=(0, (4, 3)))
            for i, (name, who) in enumerate(groups.items()):
                g = gains(cell, who, phase, column)
                m, lo, hi = mean_ci(g)
                y = len(groups) - 1 - i
                colour = T["walk"] if lo > 0 else T["soft"]
                ax.plot([lo, hi], [y, y], color=colour, lw=2.4, solid_capstyle="round")
                ax.plot(m, y, "o", color=colour, ms=7)
                ax.text(max(hi, 0) + 0.4, y, f"{m:+.1f}  {pstr(p_of(g))}", color=colour, va="center", fontsize=9)
            ax.set_yticks(range(len(groups)), [f"{n} ({len(w)})" for n, w in groups.items()][::-1])
            ax.set_title(f"{PHASES[phase]} · {title}")
            ax.set_xlim(-4, 9)
            if r == 1:
                ax.set_xlabel("Walk-down minus same-volume ball (points)")
    headline(fig, "At the same volume, the tract shape catches more FLAIR recurrence, not more enhancing",
             f"Mean per-patient difference at {WALK_MM} mm, 95% bootstrap interval; blue = interval above zero")
    note(fig, "Enhancing recurrence sits within a few mm of the tumor (chart 1), so any target of the same size "
              "catches it; the shape only matters for the spread further out.")
    save(fig, folder, "05_gain_by_cohort")


def f05_robust(folder, cell):
    upenn = [p for p in patients_of(cell) if cohort_of(int(p)) == "UPenn"]
    score = {p: num(cell[(p, "46", WALK_MM, "walk")]["pathology"]) for p in upenn if (p, "46", WALK_MM, "walk") in cell}
    rows = [("UPenn, all", upenn),
            ("Without pseudoprogression (score 3-6)", [p for p in upenn if (score.get(p) or 0) >= 3]),
            ("True progression only (score 5-6)", [p for p in upenn if (score.get(p) or 0) >= 5])]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axvline(0, color=T["faint"], lw=1, ls=(0, (4, 3)))
    items = [(n, gains(cell, w, "46", "WT_beyond_pct")) for n, w in rows]
    for name, file in (("vs a tissue-matched ball, UPenn", "tissue_control.csv"),
                       ("vs a tissue-matched ball, TUM + LUMIERE", "tissue_control_tum_lumiere.csv")):
        with open(results_dir("native", RESCORED) / file) as fh:
            tc = list(csv.DictReader(fh))
        items.append((name, np.array([float(r["target WT beyond"]) - float(r["tissue matched ball WT beyond"])
                                      for r in tc])))
    for i, (name, g) in enumerate(items):
        m, lo, hi = mean_ci(g)
        y = len(items) - 1 - i
        colour = T["walk"] if lo > 0 else T["soft"]
        ax.plot([lo, hi], [y, y], color=colour, lw=2.6, solid_capstyle="round")
        ax.plot(m, y, "o", color=colour, ms=8)
        ax.text(hi + 0.3, y, f"{m:+.1f}  {pstr(p_of(g))}  ({len(g)})", color=colour, va="center", fontsize=10)
    ax.axhline(1.5, color=T["rule"], lw=1)
    ax.set_yticks(range(len(items)), [n for n, _ in items][::-1])
    ax.set_xlabel("Walk-down minus comparison ball, FLAIR recurrence beyond the tumor (points)")
    ax.set_xlim(-3, 9)
    headline(fig, "The gain holds without pseudoprogression, and partly survives a white-matter-matched control",
             "46 Gy, recurrence outside the original tumor, mean and 95% bootstrap interval")
    note(fig, "Pathology score = share of second-surgery tissue that was tumor (UPenn). The tissue-matched ball has "
              "the walk-down's white-matter share, so what remains is the shape, not just the tissue.")
    save(fig, folder, "06_robustness")


# ---------- part 2: patient diffusion ----------

def f06_coverage(folder):
    with open(results_dir("native") / "coverage_by_distance.csv") as fh:
        rows = list(csv.reader(fh))[1:]
    mm = np.arange(0, 31)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for part, label, colour in (("enhancing", "around the enhancing tumor", T["walk"]),
                                ("non-enhancing", "around the non-enhancing tumor and edema", T["other"])):
        Y = np.array([[float(v) for v in r[2:]] for r in rows if r[1] == part]) * 100
        lo, mid, hi = (np.nanpercentile(Y, q, 0) for q in (25, 50, 75))
        ax.fill_between(mm, lo, hi, color=colour, alpha=0.15, lw=0)
        ax.plot(mm, mid, color=colour, lw=2.4)
        ax.text(30.4, mid[-1], label, color=colour, fontsize=9, va="center")
        full = int((Y[:, 0] >= 99.995).sum())
    ax.set_xlim(0, 30)
    ax.set_ylim(80, 100.5)
    ax.set_xlabel("Distance from the tumor (mm), 0 = inside it")
    ax.set_ylabel("Brain the diffusion scan reached (%)")
    headline(fig, "The diffusion scan covers the tumor and the area around it",
             "UPenn's diffusion is a 12 cm slab; what it misses is mostly far from the tumor")
    note(fig, "Line = median of 38 pre-op scans, band = middle 50%. Only 00307 (70% of its enhancing tumor), "
              "00086 and 00122 miss part of the tumor itself. Missing voxels are left out, never filled in.")
    save(fig, folder, "07_diffusion_coverage")


def f07_patient_vs_atlas(folder, patient, atlas):
    who = sorted(set(patients_of(patient)) & set(patients_of(atlas)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey="row")
    for r, phase in enumerate(PHASES):
        for c, (column, title) in enumerate(KINDS.items()):
            ax = axes[r, c]
            a, b = gains(patient, who, phase, column), gains(atlas, who, phase, column)
            for x, y in zip(a, b):
                ax.plot([0, 1], [x, y], color=T["patient"], lw=0.6)
            for x, v, colour in ((0, a, T["other"]), (1, b, T["walk"])):
                m = np.median(v)
                ax.plot([x - 0.14, x + 0.14], [m, m], color=colour, lw=3.5, solid_capstyle="round")
                ax.text(x + (0.2 if x else -0.2), m, f"{m:+.1f}", color=colour, fontsize=11, fontweight="bold",
                        va="center", ha="left" if x else "right")
            ax.axhline(0, color=T["faint"], lw=1, ls=(0, (4, 3)))
            ax.set_title(f"{PHASES[phase]} · {title}   atlas minus patient {np.median(b - a):+.1f}, {pstr(p_of(b - a))}",
                         fontsize=10)
            ax.set_xticks([0, 1], ["Patient tracts", "Atlas tracts"])
            ax.set_xlim(-0.7, 1.7)
            if c == 0:
                ax.set_ylabel("Walk-down minus ball (points)")
            lim = np.nanpercentile(np.abs(np.r_[a, b]), 97) * 1.2
            ax.set_ylim(-lim, lim)
    headline(fig, "Walking along the patient's own tracts gives no gain; the atlas does",
             f"Same {len(who)} UPenn patients, both scored only where the diffusion scan reached")
    note(fig, "Grey line = patient, thick tick = median. With a strict coverage cut (34 patients with 99%+ of the "
              "tumor and 90%+ of the 20 mm around it covered) the 46 Gy FLAIR result is the same: +1.4, p < 0.001. "
              "Axes clipped at the 97th percentile.")
    save(fig, folder, "08_patient_vs_atlas_tracts")


def f08_distance(folder, patient, atlas):
    who = sorted(set(patients_of(patient)) & set(patients_of(atlas)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey=True)
    for r, phase in enumerate(PHASES):
        for c, (column, title) in enumerate(KINDS.items()):
            ax = axes[r, c]
            for cell, label, colour in ((patient, "Patient tracts", T["other"]), (atlas, "Atlas tracts", T["walk"])):
                Y = np.array([[np.nan if not len(g := gains(cell, [p], phase, column, mm)) else g[0]
                               for mm in range(41)] for p in who])
                mid = np.nanmedian(Y, 0)
                ax.plot(range(41), mid, color=colour, lw=2.4)
                # explainer: label once, in the top-left panel, atlas above its line and patient below
                if r == 0 and c == 0:
                    at = 10 if colour == T["walk"] else 30
                    ax.text(at, mid[at] + (0.3 if colour == T["walk"] else -0.3), label, color=colour, fontsize=10,
                            ha="right" if colour == T["walk"] else "center",
                            va="bottom" if colour == T["walk"] else "top")
            ax.axhline(0, color=T["faint"], lw=1, ls=(0, (4, 3)))
            ax.axvline(WALK_MM, color=T["rule"], lw=1)
            ax.set_title(f"{PHASES[phase]} · {title}")
            ax.set_xlim(0, 40)
            if r == 1:
                ax.set_xlabel("Walk distance along the tracts (mm)")
            if c == 0:
                ax.set_ylabel("Walk-down minus ball,\nmedian (points)")
    # explainer: shared axis, so the range is set once after every panel has drawn
    axes[0, 0].set_ylim(-0.9, max(a.get_ylim()[1] for a in axes.flat) + 0.3)
    headline(fig, "No walk distance rescues the patient's own tracts",
             "Median gain over the same-volume ball at every walk distance; our walk is 20 mm")
    note(fig, "Enhancing recurrence stays within a few mm of the tumor, so neither tract source changes how much of it "
              "a same-size target catches.\nLikely cause for the FLAIR gap: the tumor pushes fibers aside (mass effect), "
              "so pre-op tracts wrap around it instead of leading away, and tracking fails in edema.")
    save(fig, folder, "09_walk_distance")


def f09_deviation(folder):
    with open(results_dir("native") / "deviation_auc_upenn.csv") as fh:
        rows = list(csv.DictReader(fh))
    maps = (("md", "MD (ADC)"), ("tissue_md", "Tissue MD"), ("dti_fa", "FA"), ("fw", "Free water"))
    kinds = (("z", "higher than healthy", "white"), ("|z|", "deviates either way", "hollow"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for r, (comp, ctitle) in enumerate((("normal-appearing", "Normal-appearing tissue"), ("edema", "Inside pre-op edema"))):
        for c, (outcome, otitle) in enumerate((("enhancing", "becomes enhancing"), ("non-enhancing", "becomes non-enhancing"))):
            ax = axes[r, c]
            ax.axvline(0.5, color=T["faint"], lw=1, ls=(0, (4, 3)))
            ticks, labels = [], []
            for i, (m, mname) in enumerate(maps):
                for j, (k, kname, fill) in enumerate(kinds):
                    hit = [x for x in rows if x["outcome"] == outcome and x["compartment"] == comp
                           and x["score"] == f"{m} {k}"]
                    if not hit:
                        continue
                    h, y = hit[0], -(i * 3 + j)
                    ax.plot([float(h["ci_low"]), float(h["ci_high"])], [y, y], color=T["walk"], lw=2)
                    ax.plot(float(h["auc"]), y, "o", color=T["walk"], ms=7,
                            mfc=T["walk"] if fill == "white" else T["bg"], mew=1.6)
                    ticks.append(y)
                    labels.append(f"{mname}, {kname}" if j == 0 else kname)
            ax.set_yticks(ticks, labels)
            ax.set_title(f"{ctitle} · {otitle}", fontsize=10)
            ax.set_xlim(0.35, 0.7)
            if r == 1:
                ax.set_xlabel("Separates recurring from stable tissue at the same distance (AUC)")
    headline(fig, "Diffusion barely tells recurring tissue from stable tissue at the same distance",
             "Each voxel scored against the patient's own healthy tissue 30+ mm away; 0.5 = chance")
    note(fig, "Filled = higher than healthy, hollow = deviates in either direction (Reshma's idea). Bars = 95% "
              "interval resampling patients. 38 UPenn patients, 4 mm distance bands.")
    save(fig, folder, "10_diffusion_vs_recurrence")


def f10_models(folder):
    with open(results_dir("native") / "deviation_model_upenn.csv") as fh:
        rows = list(csv.DictReader(fh))
    steps = (("distance only", "Distance from tumor"), ("distance + diffusion", "+ diffusion"),
             ("distance + diffusion + tissue md", "+ tissue MD"))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for k, (outcome, label) in enumerate((("enhancing", "Becomes enhancing"), ("non-enhancing", "Becomes non-enhancing"))):
        vals = [float(next(r for r in rows if r["outcome"] == outcome and r["model"] == s)["held_out_auc"])
                for s, _ in steps]
        ys = [-(k * 4 + i) for i in range(3)]
        ax.plot(vals, ys, color=T["walk"], lw=1, alpha=0.6)
        for v, y, (_, name) in zip(vals, ys, steps):
            ax.plot(v, y, "o", color=T["walk"], ms=8)
            ax.text(v + 0.006, y, f"{v:.3f}", color=T["soft"], va="center", fontsize=10)
            ax.text(0.495, y, name, color=T["soft"], va="center", ha="right", fontsize=10)
        ax.text(0.495, ys[0] + 0.9, label, color=T["ink"], ha="right", fontsize=11, fontweight="bold")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-7.5, 1.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Held-out AUC, leave one patient out")
    headline(fig, "Distance does the work; diffusion adds nothing on top",
             "Predicting which tissue recurs, trained on all patients but one and tested on that one")
    save(fig, folder, "11_distance_vs_diffusion")


def main(theme="dark"):
    global T
    T = THEMES[theme]
    style()
    folder = results_dir("native").parent / "figures" / "lab" / theme
    folder.mkdir(parents=True, exist_ok=True)
    atlas_all = curves("walkdown_curves.csv", "walkdown_tum_lumiere_curves.csv")
    patient = curves("walkdown_dtiprep_covered_curves.csv")
    atlas_cov = curves("walkdown_covered_curves.csv")
    f01_reach(folder)
    f02_location(folder, theme)
    f02_brain(folder, atlas_all)
    f03_equal_volume(folder, atlas_all)
    f04_by_cohort(folder, atlas_all)
    f05_robust(folder, atlas_all)
    f06_coverage(folder)
    f07_patient_vs_atlas(folder, patient, atlas_cov)
    f08_distance(folder, patient, atlas_cov)
    f09_deviation(folder)
    f10_models(folder)


if __name__ == "__main__":
    for theme in sys.argv[1:] or ("dark",):
        main(theme)
