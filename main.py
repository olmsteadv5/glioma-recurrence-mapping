#!/home/vilem/Projects/YBAParcelVisualizer/.venv/bin/python
# Run the whole pipeline for chosen UPenn-GBM subjects.

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
    print(subject)
    out_dir(subject).mkdir(parents=True, exist_ok=True)

    stripped = skullstrip.run(subject)
    verify.run(subject, stripped)

    seg = segment.run(subject, stripped)
    scores = evaluate.run(subject, seg)
    register.run(subject, stripped, seg)

    # printed here because the segmenter's progress spinner erases the line above it
    r = scores["regions_ml"]
    print(f"  WT {r['WT']} mL, TC {r['TC']} mL, ET {r['ET']} mL, "
          f"{time.time() - started:.0f}s")
    if "dice" in scores:
        d = scores["dice"]
        print(f"  dice WT {d['WT']}, TC {d['TC']}, ET {d['ET']}")
    return subject, scores


def main(numbers):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    results = {}
    for number in numbers:
        subject, scores = process(number)
        results[subject] = scores

    labels, names = atlas.label_volume(nib.load(MNI))

    for subject in results:
        overlap = atlas.overlap(out_dir(subject) / "seg_in_mni.nii.gz", labels, names)
        results[subject]["atlas"] = overlap
        print(f"{subject}: {overlap['in_parcels_ml']} mL in "
              f"{len(overlap['parcels'])} parcels")

    # merge so running one subject doesn't wipe the others
    stored = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    stored.update(results)
    RESULTS.write_text(json.dumps(stored, indent=2))
    print(f"wrote {RESULTS}")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python main.py <subject numbers>   e.g. python main.py 36 42 51")
    main([int(a) for a in sys.argv[1:]])
    print("open results.ipynb to look at them")
