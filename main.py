#!/usr/bin/env python3
"""Run the whole pipeline for chosen UPenn-GBM subjects.

    python main.py 36 42 51
"""

import json
import sys
import time

import nibabel as nib

import atlas
import evaluate
import fetch
import register
import segment
import skullstrip
import verify
from paths import MNI, OUTPUTS, RESULTS, out_dir


def process(number):
    started = time.time()
    subject = fetch.get(number)
    print(f"[{subject}]")
    out_dir(subject).mkdir(parents=True, exist_ok=True)

    stripped = skullstrip.run(subject)
    verify.run(subject, stripped)
    print("    stripped and verified")

    seg = segment.run(subject, stripped)
    scores = evaluate.run(subject, seg)
    regions = scores["regions_ml"]
    print(f"    segmented: WT {regions['WT']} mL, TC {regions['TC']} mL, "
          f"ET {regions['ET']} mL")
    if "dice" in scores:
        dice = scores["dice"]
        print(f"    dice vs reference: " + "  ".join(
            f"{k} {v:.3f}" for k, v in dice.items() if v is not None))

    register.run(subject, stripped, seg)
    print(f"    registered to MNI ({time.time() - started:.0f}s total)")
    return subject, scores


def main(numbers):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    results = {}
    for number in numbers:
        subject, scores = process(number)
        results[subject] = scores

    print("\n[atlas]")
    yba = atlas.load_atlas()
    labels, names = atlas.label_volume(yba, nib.load(MNI))

    for subject in results:
        overlap = atlas.overlap(out_dir(subject) / "seg_in_mni.nii.gz", labels, names)
        results[subject]["atlas"] = overlap
        print(f"    {subject}: {overlap['in_parcels_ml']:.1f} of "
              f"{overlap['tumour_ml']:.1f} mL in {len(overlap['parcels'])} parcels")

    # merge, so running one subject does not discard the others already on disk
    stored = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    stored.update(results)
    RESULTS.write_text(json.dumps(stored, indent=2))
    print(f"\nWrote {RESULTS} ({len(stored)} subjects)")
    return results, yba


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python main.py <subject numbers>   e.g. python main.py 36 42 51")
    results, yba = main([int(a) for a in sys.argv[1:]])

    import view
    for subject in results:
        view.run(subject, yba, results[subject]["atlas"])
