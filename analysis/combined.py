#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: walk-down against the NRG 2 cm volume and a same-size ball, each cohort on its own first and then pooled

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.walkdown import RESCORED, WALK_MM
from paths import cohort_of, patient_key, results_dir

TABLES = {"UPenn": "walkdown_curves.csv", "PREDICT": "walkdown_tum_lumiere_curves.csv"}
_num = lambda v: None if v in ("", "None", None) else float(v)


def load(space):
    rows = []
    for name in TABLES.values():
        path = results_dir(space, RESCORED) / name
        if not path.exists():
            print(f"missing {path}, pooled is then only what is present")
        else:
            with open(path) as fh:
                rows += [dict(r, cohort=cohort_of(int(r["patient"])), key=patient_key(int(r["patient"])))
                         for r in csv.DictReader(fh)]
    # decision: one person counted once, pooled tables refuse a patient that arrives through two cohorts
    owners = {}
    for r in rows:
        owners.setdefault(r["key"], set()).add(r["cohort"])
    twice = [k for k, c in owners.items() if len(c) > 1]
    if twice:
        raise RuntimeError(f"patients in more than one cohort: {', '.join(twice[:5])}")
    return {(r["patient"], r["phase"], int(r["walk_mm"]), r["target"]): r for r in rows}, rows


def paired(cell, patients, phase, column):
    d = np.array([_num(cell[(p, phase, WALK_MM, "walk")][column]) - _num(cell[(p, phase, WALK_MM, "ball")][column])
                  for p in patients if _num(cell[(p, phase, WALK_MM, "walk")][column]) is not None
                  and _num(cell[(p, phase, WALK_MM, "ball")][column]) is not None])
    if not len(d):
        return {}
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return {f"{column} walk minus ball": round(float(np.mean(d)), 2), f"{column} better": int((d > 0.05).sum()),
            f"{column} worse": int((d < -0.05).sum()), f"{column} p": float(f"{p:.2g}")}


def locality():
    # explainer: margin around the original tumor that holds 95% of enhancing recurrence, by pathology group
    from analysis import measure
    by = {}
    for first, later in measure.pairs("native"):
        found = measure.overlap(first, later, "ET", "native")
        score = measure.pathology(first)
        if "margin_95_mm" in found and score is not None:
            for label, floor in (("all", 1), ("score 3-6", 3), ("score 5-6", 5)):
                if score >= floor:
                    by.setdefault(label, []).append(found["margin_95_mm"])
        measure.forget()
    for label, values in by.items():
        print(f"enhancing recurrence, 95% within (median mm) {label:10s} n={len(values):2d}: {np.median(values):.1f}")


def main():
    out = []
    for space in ("native", "mni"):
        cell, rows = load(space)
        if not rows:
            continue
        groups = {c: sorted({r["patient"] for r in rows if r["cohort"] == c}) for c in ("UPenn", "TUM", "LUMIERE")}
        groups = {c: p for c, p in groups.items() if p}
        # decision: UPenn's follow-up is pathology scored, scores 1-2 are mostly treatment effect, so results are also shown without them and on true progression alone
        score = {r["patient"]: _num(r["pathology"]) for r in rows if r["cohort"] == "UPenn"}
        for label, floor in (("UPenn score 3-6", 3), ("UPenn score 5-6", 5)):
            chosen = sorted(p for p, v in score.items() if v is not None and v >= floor)
            if chosen:
                groups[label] = chosen
        groups["pooled"] = sorted({r["patient"] for r in rows})
        for name, patients in groups.items():
            for phase in ("46", "60"):
                median = lambda target, column: np.median([v for v in (_num(cell[(p, phase, WALK_MM, target)][column])
                                                                       for p in patients) if v is not None])
                row = {"space": space, "cohort": name, "patients": len(patients), "phase": phase,
                       "nrg beyond tumor mL": round(float(median("iso", "beyond_gtv_ml"))),
                       "walk beyond tumor mL": round(float(median("walk", "beyond_gtv_ml")))}
                for column in ("ET_pct", "WT_pct", "ET_beyond_pct", "WT_beyond_pct"):
                    for target in ("iso", "walk", "ball"):
                        row[f"{target} {column}"] = round(float(median(target, column)), 1)
                    row.update(paired(cell, patients, phase, column))
                out.append(row)
    path = results_dir("native", RESCORED) / "walkdown_by_cohort.csv"
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(out)
    print(path)
    for r in out:
        if r["space"] == "native":
            print(f"{r['cohort']:8s} n={r['patients']:3d} {r['phase']} Gy | WT walk-ball {r.get('WT_pct walk minus ball', '')} "
                  f"(p={r.get('WT_pct p', '')}) | beyond-tumor WT walk-ball {r.get('WT_beyond_pct walk minus ball', '')} "
                  f"(p={r.get('WT_beyond_pct p', '')})")


if __name__ == "__main__":
    main()
    locality()
