#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: one row per scan, which SynthSeg structures are missing in each tissue map and how sure SynthSeg was, with flags worth a look

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import SCANS, results_dir

# explainer: SynthSeg's default structures, the cortex pair (3, 42) is replaced by 68 parcels in a --parc map
STRUCTURES = {2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28,
              41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60}
CORTEX = {3, 42}
# decision: SynthSeg's qc scores run 0 to 1, a group under this gets a look, not a verdict. our number
QC_FLOOR = 0.6
# decision: a brain this much smaller or larger in one map than the other means one of them failed
VOLUME_GAP = 0.15


def labels(path):
    data = np.asarray(nib.load(path).dataobj)
    present = set(np.unique(data).astype(int).tolist())
    # explainer: parcels stand in for the cortex label on their side
    if any(1000 <= v < 2000 for v in present):
        present.add(3)
    if any(v >= 2000 for v in present):
        present.add(42)
    zooms = nib.load(path).header.get_zooms()[:3]
    return present, float((data > 0).sum() * np.prod(zooms) / 1000)


def qc_scores(path):
    if not path.exists():
        return {}
    with open(path) as fh:
        row = next(csv.DictReader(fh), {})
    return {k: float(v) for k, v in row.items() if k != "subject" and v not in ("", None)}


def row(scan):
    here = SCANS / scan
    old, old_ml = labels(here / "tissue.nii.gz")
    out = {"scan": scan, "missing_first": " ".join(map(str, sorted(STRUCTURES - old))), "brain_ml_first": round(old_ml)}
    robust = here / "tissue_robust.nii.gz"
    flags = []
    if robust.exists():
        new, new_ml = labels(robust)
        scores = qc_scores(here / "tissue_robust_qc.csv")
        low = sorted(k for k, v in scores.items() if v < QC_FLOOR)
        out.update({"missing_robust": " ".join(map(str, sorted(STRUCTURES - new))), "brain_ml_robust": round(new_ml),
                    "qc_lowest": round(min(scores.values()), 3) if scores else None, "qc_low_groups": "; ".join(low)})
        if STRUCTURES - new:
            flags.append("robust map missing structures")
        if low:
            flags.append("low qc score")
        if abs(new_ml - old_ml) > VOLUME_GAP * max(new_ml, old_ml):
            flags.append("brain volume differs between maps")
    else:
        out.update({"missing_robust": None, "brain_ml_robust": None, "qc_lowest": None, "qc_low_groups": None})
        flags.append("no robust map yet")
    if STRUCTURES - old:
        flags.append("first map missing structures")
    out["flags"] = "; ".join(flags)
    return out


def main():
    scans = sorted(p.name for p in SCANS.iterdir() if (p / "tissue.nii.gz").exists())
    rows = [row(s) for s in scans]
    path = results_dir("native") / "tissue_qc.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)
    cohorts = sorted({s.rsplit("-", 1)[0] for s in scans})
    for cohort in cohorts:
        mine = [r for r in rows if r["scan"].startswith(cohort)]
        done = [r for r in mine if r["missing_robust"] is not None]
        print(f"{cohort:10s} {len(mine):4d} scans | first map missing structures {sum(bool(r['missing_first']) for r in mine):3d}"
              f" | robust done {len(done):3d}, missing structures {sum(bool(r['missing_robust']) for r in done):3d},"
              f" low qc {sum('low qc' in r['flags'] for r in done):3d}")


if __name__ == "__main__":
    main()
