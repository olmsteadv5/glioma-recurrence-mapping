#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: fails when a results table counts one person under two cohorts, or includes a cohort left out of analysis

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import ANALYSIS, cohort_of, in_analysis, patient_key


def problems(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "patient" not in rows[0]:
        return []
    numbers = {int(r["patient"]) for r in rows if r["patient"].isdigit()}
    seen, out = {}, []
    for n in sorted(numbers):
        seen.setdefault(patient_key(n), set()).add(cohort_of(n))
    out += [f"{key} appears as {', '.join(sorted(cohorts))}" for key, cohorts in seen.items() if len(cohorts) > 1]
    left = sorted({cohort_of(n) for n in numbers if not in_analysis(n)})
    out += [f"includes {name}, which is left out of analysis" for name in left]
    return out


def main(paths):
    failed = False
    for path in paths or sorted(ANALYSIS.rglob("*.csv")):
        found = problems(path)
        failed |= bool(found)
        print(f"{path}: {'ok' if not found else 'FAIL'}")
        for line in found:
            print(f"  {line}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
