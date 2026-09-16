#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Run the whole pipeline for chosen UPenn-GBM subjects.

import json
import sys
import time

import nibabel as nib

import atlas
import evaluate
import function
import fetch
import register
import segment
import skullstrip
import verify
from paths import (MNI, OUTPUTS, RESULTS, modality, out_dir, raw_dir,
                   stripped_dir, subject_id)


STAGES = ("fetch", "strip", "segment", "register", "map")
NEEDS = {"strip": "inputs", "segment": "stripped scans", "register": "a segmentation",
         "map": "a registered segmentation"}


def process(number, first="fetch"):
    # Stages are sequential, so starting at one means running all the later ones.
    started = time.time()
    at = STAGES.index(first)
    subject = fetch.get(number) if at == 0 else subject_id(number)
    print(subject)
    out_dir(subject).mkdir(parents=True, exist_ok=True)

    stripped = stripped_dir(subject)
    if at <= 1:
        stripped = skullstrip.run(subject)
        verify.run(subject, stripped)

    seg = out_dir(subject) / "seg.nii.gz"
    if at <= 2:
        seg = segment.run(subject, stripped)
    scores = evaluate.run(subject, seg)

    if at <= 3:
        register.run(subject, stripped, seg)

    # printed here because the segmenter's progress spinner erases the line above it
    r = scores["regions_ml"]
    print(f"  WT {r['WT']} mL, TC {r['TC']} mL, ET {r['ET']} mL, "
          f"{time.time() - started:.0f}s")
    if "dice" in scores:
        d = scores["dice"]
        print(f"  dice WT {d['WT']}, TC {d['TC']}, ET {d['ET']}")
    return subject, scores


def missing(number, first):
    # What a later starting point assumes is already on disk.
    subject = subject_id(number)
    at = STAGES.index(first)
    if at >= 1 and not raw_dir(subject).exists():
        return f"{raw_dir(subject)} is missing, start from fetch"
    if at >= 2 and not modality(stripped_dir(subject), subject, "T1").exists():
        return f"{stripped_dir(subject)} has no T1, start from strip"
    if at >= 3 and not (out_dir(subject) / "seg.nii.gz").exists():
        return f"{subject} has no segmentation, start from segment"
    if at >= 4 and not (out_dir(subject) / "seg_in_mni.nii.gz").exists():
        return f"{subject} is not registered, start from register"
    return None


def main(numbers, first="fetch"):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stored = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results = {}
    for number in numbers:
        gap = missing(number, first)
        if gap:
            sys.exit(gap)
        if first == "map":
            subject = subject_id(number)
            results[subject] = stored.get(subject, {})
            results[subject].update(evaluate.run(subject, out_dir(subject) / "seg.nii.gz"))
        else:
            subject, scores = process(number, first)
            results[subject] = scores

    labels, names = atlas.label_volume(nib.load(MNI))

    for subject in results:
        overlap = atlas.overlap(out_dir(subject) / "seg_in_mni.nii.gz", labels, names)
        results[subject]["atlas"] = overlap
        results[subject]["function"] = function.for_tumour(overlap)
        top = ", ".join(f["term"] for f in results[subject]["function"][:4])
        print(f"{subject}: {overlap['in_parcels_ml']} mL in "
              f"{len(overlap['parcels'])} parcels; {top}")

    # merge so running one subject doesn't wipe the others
    stored.update(results)
    RESULTS.write_text(json.dumps(stored, indent=2))
    print(f"wrote {RESULTS}")
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    # the terminal UI is its own viewer, it passes --no-view
    show = "--no-view" not in args
    args = [a for a in args if a != "--no-view"]
    first = "fetch"
    if "--from" in args:
        cut = args.index("--from")
        first = args[cut + 1] if cut + 1 < len(args) else ""
        args = args[:cut] + args[cut + 2:]
    if not args or not all(a.isdigit() for a in args) or first not in STAGES:
        sys.exit(f"usage: ./main.py <subject numbers> [--from {'|'.join(STAGES)}] "
                 f"[--no-view]")

    results = main([int(a) for a in args], first)

    if show:
        import view
        for subject in results:
            view.run(subject, results[subject]["atlas"])
