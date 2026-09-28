#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# One row per patient: how much of the later tumor sits on, and near, the first.

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from paths import OUTPUTS

PARTS = (("ET", "enhancing"), ("TC", "core"), ("WT", "whole"))


def rows(review):
    by = {}
    for x in review:
        by.setdefault(x["patient"], {"days": x["days"]})[x["region"]] = x
    for patient in sorted(by):
        row = [patient, by[patient]["days"]]
        for code, _ in PARTS:
            x = by[patient].get(code, {})
            if "inside" in x:
                row += [round(x["first_ml"], 1), round(x["later_ml"], 1), round(100 * x["inside"], 1),
                        round(100 * x["in_4600"], 1), x["pattern"]]
            else:
                # nothing of this part came back, so there is nothing to place
                row += [round(x.get("first_ml", 0), 1), 0.0, "", "", "none at follow-up"]
        yield row


def main(path=OUTPUTS / "overlap_review.csv"):
    head = ["patient", "days_between_scans"]
    for _, name in PARTS:
        head += [f"{name}_first_ml", f"{name}_later_ml", f"{name}_pct_on_original",
                 f"{name}_pct_inside_2cm_plan", f"{name}_pattern"]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(head)
        writer.writerows(rows(measure.review(measure.pairs())))
    return path


if __name__ == "__main__":
    print(main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUTS / "overlap_review.csv"))
