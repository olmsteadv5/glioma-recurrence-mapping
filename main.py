#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Run the whole pipeline for chosen UPenn-GBM subjects.

import json
import sys
import time

import nibabel as nib

from scan import align, evaluate, fetch, register, segment, skullstrip, tissue, verify
from yale import atlas, function
from paths import (MNI, OUTPUTS, RESULTS, STAGES, modality, out_dir, raw_dir,
                   stripped_dir, subject_id)


def mark(stage):
    # the terminal UI reads these to follow along, they are not chatter
    print(f"* {stage}", flush=True)


def process(number, first="fetch", timepoint=1, was=None):
    # Stages are sequential, so starting at one means running all the later ones.
    started = time.time()
    at = STAGES.index(first)
    subject = subject_id(number, timepoint)
    run = dict(was or {})
    times = dict(run.get("times", {}))

    def took(stage, since):
        times[stage] = round(time.time() - since, 1)

    if at == 0:
        mark("fetch")
        clock = time.time()
        subject = fetch.get(number, timepoint)
        took("fetch", clock)
    print(subject)
    out_dir(subject).mkdir(parents=True, exist_ok=True)

    stripped = stripped_dir(subject)
    if at <= 1:
        mark("strip")
        clock = time.time()
        stripped = skullstrip.run(subject)
        verify.run(subject, stripped)
        took("strip", clock)

    if at <= 2:
        mark("tissue")
        clock = time.time()
        tissue.run(subject)
        took("tissue", clock)

    seg = out_dir(subject) / "seg.nii.gz"
    if at <= 3:
        mark("segment")
        clock = time.time()
        seg, run["algorithm"], run["device"] = segment.run(subject, stripped)
        took("segment", clock)
    scores = evaluate.run(subject, seg)

    if at <= 4:
        mark("register")
        clock = time.time()
        _, run["kept"] = register.run(subject, stripped, seg)
        # the reslice needs the warp, so it cannot happen back at the tissue stage
        tissue.to_mni(subject)
        took("register", clock)

    if at <= 5 and timepoint == 2:
        mark("align")
        clock = time.time()
        align.run(subject)
        took("align", clock)
    run["times"] = times
    run["finished"], run["seconds"] = time.time(), round(time.time() - started)
    scores["run"] = run

    # printed here because the segmenter's progress spinner erases the line above it
    r = scores["regions_ml"]
    print(f"  WT {r['WT']} mL, TC {r['TC']} mL, ET {r['ET']} mL, "
          f"{time.time() - started:.0f}s")
    if "dice" in scores:
        d = scores["dice"]
        print(f"  dice WT {d['WT']}, TC {d['TC']}, ET {d['ET']}")
    return subject, scores


def missing(number, first, timepoint=1):
    # What a later starting point assumes is already on disk.
    subject = subject_id(number, timepoint)
    at = STAGES.index(first)
    if at >= 1 and not raw_dir(subject).exists():
        return f"{raw_dir(subject)} is missing, start from fetch"
    if at >= 2 and not modality(stripped_dir(subject), subject, "T1").exists():
        return f"{stripped_dir(subject)} has no T1, start from strip"
    # tissue is deliberately not required here; the 26 subjects segmented before
    # it existed must still reach map
    if at >= 4 and not (out_dir(subject) / "seg.nii.gz").exists():
        return f"{subject} has no segmentation, start from segment"
    if at >= 6 and not (out_dir(subject) / "seg_in_mni.nii.gz").exists():
        return f"{subject} is not registered, start from register"
    if at >= 5 and timepoint == 2 and not stripped_dir(subject_id(number, 1)).exists():
        return f"{subject_id(number, 1)} has no pre-op scan to align, run it first"
    return None


def main(numbers, first="fetch", timepoint=1):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stored = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results = {}
    for number in numbers:
        gap = missing(number, first, timepoint)
        if gap:
            sys.exit(gap)
        subject = subject_id(number, timepoint)
        was = stored.get(subject, {}).get("run")
        if first == "map":
            results[subject] = stored.get(subject, {})
            results[subject].update(evaluate.run(subject, out_dir(subject) / "seg.nii.gz"))
        else:
            subject, scores = process(number, first, timepoint, was)
            results[subject] = scores

    mark("map")
    clock = time.time()
    labels, names = atlas.label_volume(nib.load(MNI))

    for subject in results:
        overlap = atlas.overlap(out_dir(subject) / "seg_in_mni.nii.gz", labels, names)
        results[subject]["atlas"] = overlap
        results[subject]["function"] = function.for_tumor(overlap)
        block = results[subject].setdefault("run", {})
        block["mapped"] = time.time()
        block.setdefault("times", {})["map"] = round(time.time() - clock, 1)
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
    first, timepoint = "fetch", "1"
    for flag, fallback in (("--from", "fetch"), ("--timepoint", "1")):
        if flag in args:
            cut = args.index(flag)
            value = args[cut + 1] if cut + 1 < len(args) else ""
            args = args[:cut] + args[cut + 2:]
            if flag == "--from":
                first = value
            else:
                timepoint = value
    if (not args or not all(a.isdigit() for a in args)
            or first not in STAGES or timepoint not in ("1", "2")):
        sys.exit(f"usage: ./main.py <subject numbers> [--from {'|'.join(STAGES)}] "
                 f"[--timepoint 1|2] [--no-view]")

    results = main([int(a) for a in args], first, int(timepoint))

    if show:
        from show import view
        for subject in results:
            view.run(subject, results[subject]["atlas"])
