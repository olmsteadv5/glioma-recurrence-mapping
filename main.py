#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python

import fcntl
import json
import sys
import time

import nibabel as nib

from scan import align, evaluate, fetch, ingest, register, segment, skullstrip, tissue, verify
from datasets import adapter
from yale import atlas, function
from paths import (MNI, OUTPUTS, RESULTS, STAGES, anatomy, external, out_dir, raw_dir,
                   stripped_dir, subject_id)


def mark(stage):
    # decision: the tui parses these lines, don't remove
    print(f"* {stage}", flush=True)


def process(number, first="fetch", timepoint=1, was=None, last="map"):
    # explainer: starting at a stage runs every stage after it too
    started = time.time()
    at = STAGES.index(first)
    subject = subject_id(number, timepoint)
    run = dict(was or {})
    times = dict(run.get("times", {}))
    # decision: outside datasets come stripped and segmented, so fetch is an import and strip/segment are skipped
    shipped = external(number)
    on_preop_grid = shipped and adapter(number).FOLLOWUP_ON_PREOP_GRID

    def took(stage, since):
        times[stage] = round(time.time() - since, 1)

    if at == 0:
        mark("fetch")
        clock = time.time()
        if shipped:
            subject, run["labels"] = ingest.run(number, timepoint)
        elif number >= 10000:
            subject = adapter(number).get(number, timepoint)
        else:
            subject = fetch.get(number, timepoint)
        took("fetch", clock)
    print(subject)
    out_dir(subject).mkdir(parents=True, exist_ok=True)

    stripped = stripped_dir(subject)
    if at <= 1 and not shipped:
        mark("strip")
        clock = time.time()
        if number < 10000 or not getattr(adapter(number), "PRESTRIPPED", False):
            stripped = skullstrip.run(subject)
        verify.run(subject, stripped)
        took("strip", clock)

    if at <= 2:
        mark("tissue")
        clock = time.time()
        tissue.run(subject, fast=shipped or (number >= 10000 and getattr(adapter(number), "FAST_SYNTHSEG", False)))
        took("tissue", clock)
    # decision: stop here when segmentation runs elsewhere, e.g. on a cluster gpu
    if last == "tissue":
        return subject, None

    seg = out_dir(subject) / "seg.nii.gz"
    if at <= 3 and shipped:
        run["algorithm"] = f"{adapter(number).__name__.split('.')[-1]}, {run.get('labels', 'auto')} labels"
    elif at <= 3:
        mark("segment")
        clock = time.time()
        seg, run["algorithm"], run["device"] = segment.run(subject, stripped)
        took("segment", clock)
    scores = evaluate.run(subject, seg)

    if at <= 4:
        mark("register")
        clock = time.time()
        if on_preop_grid and timepoint == 2:
            _, run["kept"] = ingest.reuse_registration(subject, seg)
        else:
            _, run["kept"] = register.run(subject, stripped, seg)
        # explainer: reslice needs the warp, so it can't run in the tissue stage
        tissue.to_mni(subject)
        took("register", clock)

    if at <= 5 and timepoint == 2:
        mark("align")
        clock = time.time()
        ingest.identity_align(subject) if on_preop_grid else align.run(subject)
        took("align", clock)
    run["times"] = times
    run["finished"], run["seconds"] = time.time(), round(time.time() - started)
    scores["run"] = run

    # decision: print here, the segmenter's spinner overwrites the line above
    r = scores["regions_ml"]
    print(f"  WT {r['WT']} mL, TC {r['TC']} mL, ET {r['ET']} mL, "
          f"{time.time() - started:.0f}s")
    if "dice" in scores:
        d = scores["dice"]
        print(f"  dice WT {d['WT']}, TC {d['TC']}, ET {d['ET']}")
    return subject, scores


def missing(number, first, timepoint=1):
    subject = subject_id(number, timepoint)
    at = STAGES.index(first)
    if at >= 1 and not raw_dir(subject).exists():
        return f"{raw_dir(subject)} is missing, start from fetch"
    if at >= 2 and not anatomy(stripped_dir(subject), subject).exists():
        return f"{stripped_dir(subject)} has no T1, start from strip"
    # decision: tissue isn't required, scans segmented before that stage existed still need to reach map
    if at >= 4 and not (out_dir(subject) / "seg.nii.gz").exists():
        return f"{subject} has no segmentation, start from segment"
    if at >= 6 and not (out_dir(subject) / "seg_in_mni.nii.gz").exists():
        return f"{subject} is not registered, start from register"
    if at >= 5 and timepoint == 2 and not stripped_dir(subject_id(number, 1)).exists():
        return f"{subject_id(number, 1)} has no pre-op scan to align, run it first"
    return None


def main(numbers, first="fetch", timepoint=1, last="map"):
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
            subject, scores = process(number, first, timepoint, was, last)
            results[subject] = scores
    if last != "map":
        return results

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
        print(f"{subject}: {overlap['in_parcels_ml']} mL in {len(overlap['parcels'])} parcels; {top}")

    # decision: merge so running one subject doesn't wipe the rest. re-read under a lock, parallel runs would drop each other's results
    with open(RESULTS.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stored = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        stored.update(results)
        partial = RESULTS.with_suffix(".tmp")
        partial.write_text(json.dumps(stored, indent=2))
        partial.replace(RESULTS)
    print(f"wrote {RESULTS}")
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    first, timepoint, last = "fetch", "1", "map"
    for flag in ("--from", "--timepoint", "--until"):
        if flag in args:
            cut = args.index(flag)
            value = args[cut + 1] if cut + 1 < len(args) else ""
            args = args[:cut] + args[cut + 2:]
            if flag == "--from":
                first = value
            elif flag == "--timepoint":
                timepoint = value
            else:
                last = value
    if (not args or not all(a.isdigit() for a in args)
            or first not in STAGES or timepoint not in ("1", "2") or last not in ("tissue", "map")):
        sys.exit(f"usage: ./main.py <subject numbers> [--from {'|'.join(STAGES)}] "
                 f"[--timepoint 1|2] [--until tissue]")

    main([int(a) for a in args], first, int(timepoint), last)
