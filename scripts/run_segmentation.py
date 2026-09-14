#!/usr/bin/env python3
"""Run BraTS adult-glioma segmentation over the UPenn-GBM subjects in data/.

Reads  data/PENN_GBM_nii/<SUBJECT>/<SUBJECT>_{T1,T1GD,T2,FLAIR}.nii.gz
Writes outputs/<algorithm>/<SUBJECT>_seg.nii.gz
       logs/<algorithm>/<SUBJECT>.log

Algorithm choice
----------------
BraTS25_1 (Jain et al.) is the 1st-place BraTS 2025 entry and is the default
*when a CUDA GPU is present*. It ships no CPU path, so on a CPU-only host the
runner falls back to BraTS25_2 (Qu Lin et al., 2nd place), which the upstream
package marks cpu_compatible at roughly 7 min/subject. Override with
--algorithm at any time; --require-algorithm turns the fallback into an error.

CPU-compatible entries, per upstream brats/data/meta:
    BraTS25_2   2nd place   ~7 min/subject
    BraTS25_3A  3rd (tie)   ~150 min/subject
GPU-only entries:
    BraTS25_1   1st place
    BraTS25_3B  3rd (tie)

Concurrency
-----------
Each subject runs a full nnU-Net-class container. Default is one at a time:
these images want several GB of RAM each, and oversubscribing a CPU-only host
makes the whole run slower, not faster. Raise --jobs only with headroom.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from brats import AdultGliomaPreAndPostTreatmentSegmenter
from brats.constants import AdultGliomaPreAndPostTreatmentAlgorithms as Algorithms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NII_DIR = PROJECT_ROOT / "data" / "PENN_GBM_nii"
OUT_ROOT = PROJECT_ROOT / "outputs"
LOG_ROOT = PROJECT_ROOT / "logs"

MODALITIES = ("T1", "T1GD", "T2", "FLAIR")

# Algorithms usable without a CUDA device, best first.
CPU_COMPATIBLE = (Algorithms.BraTS25_2, Algorithms.BraTS25_3A)


def cuda_available() -> bool:
    """True if nvidia-smi reports a working driver (matches brats' own check)."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True,
                              timeout=30).returncode == 0
    except Exception:
        return False


def discover_subjects(only: list[str] | None) -> list[str]:
    """Subject directories that contain all four required modalities."""
    if not NII_DIR.is_dir():
        raise SystemExit(f"No input directory: {NII_DIR}\n"
                         f"Run scripts/fetch_data.py first.")

    subjects, incomplete = [], []
    for d in sorted(p for p in NII_DIR.iterdir() if p.is_dir()):
        missing = [m for m in MODALITIES if not (d / f"{d.name}_{m}.nii.gz").exists()]
        (incomplete if missing else subjects).append(
            (d.name, missing) if missing else d.name)

    for name, missing in incomplete:
        print(f"skip {name}: missing {', '.join(missing)}", file=sys.stderr)

    if only:
        wanted = set(only)
        matched = [s for s in subjects if s in wanted or s.split("_")[0] in wanted]
        unknown = wanted - set(matched) - {s.split("_")[0] for s in matched}
        if unknown:
            raise SystemExit(f"Unknown subject(s): {', '.join(sorted(unknown))}")
        subjects = matched

    return subjects


def segment_subject(subject: str, algorithm_name: str, force_cpu: bool) -> dict:
    """Segment one subject. Runs in its own process; returns a result record."""
    algorithm = Algorithms(algorithm_name)
    subj_dir = NII_DIR / subject
    out_dir = OUT_ROOT / algorithm_name
    log_dir = LOG_ROOT / algorithm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / f"{subject}_seg.nii.gz"
    log_file = log_dir / f"{subject}.log"
    started = time.time()

    try:
        segmenter = AdultGliomaPreAndPostTreatmentSegmenter(
            algorithm=algorithm, force_cpu=force_cpu)
        segmenter.infer_single(
            t1c=subj_dir / f"{subject}_T1GD.nii.gz",   # post-contrast T1
            t1n=subj_dir / f"{subject}_T1.nii.gz",     # native T1
            t2f=subj_dir / f"{subject}_FLAIR.nii.gz",  # T2-FLAIR
            t2w=subj_dir / f"{subject}_T2.nii.gz",     # T2
            output_file=output_file,
            log_file=log_file,
        )
    except Exception as exc:
        return {"subject": subject, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "seconds": round(time.time() - started, 1),
                "log": str(log_file)}

    if not output_file.exists():
        return {"subject": subject, "status": "failed",
                "error": "algorithm reported success but wrote no output",
                "seconds": round(time.time() - started, 1),
                "log": str(log_file)}

    return {"subject": subject, "status": "ok",
            "output": str(output_file.relative_to(PROJECT_ROOT)),
            "seconds": round(time.time() - started, 1),
            "log": str(log_file)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", choices=[a.value for a in Algorithms],
                    help="default: BraTS25_1 on GPU hosts, BraTS25_2 on CPU-only")
    ap.add_argument("--require-algorithm", action="store_true",
                    help="fail instead of falling back when the choice needs a GPU")
    ap.add_argument("--subjects", nargs="*",
                    help="subject ids to run (default: everything in data/)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="containers in parallel (default 1; raise only with RAM)")
    ap.add_argument("--force", action="store_true", help="redo completed subjects")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and exit")
    args = ap.parse_args()

    gpu = cuda_available()
    # The default is always the competition winner; hardware may force a fallback
    # below, and we want that trade-off printed rather than silently applied.
    requested = Algorithms(args.algorithm) if args.algorithm else Algorithms.BraTS25_1

    # Resolve the algorithm against the hardware actually present.
    algorithm = requested
    if not gpu and requested not in CPU_COMPATIBLE:
        if args.require_algorithm:
            raise SystemExit(
                f"{requested.value} requires a CUDA GPU and none was found. "
                f"CPU-capable: {', '.join(a.value for a in CPU_COMPATIBLE)}.")
        algorithm = CPU_COMPATIBLE[0]
        print(f"!! {requested.value} needs a CUDA GPU; none found.\n"
              f"!! Falling back to {algorithm.value} "
              f"(BraTS25 2nd place, CPU-capable).\n"
              f"!! Use --require-algorithm to make this an error instead.\n")

    force_cpu = not gpu
    subjects = discover_subjects(args.subjects)

    if not args.force:
        done = [s for s in subjects
                if (OUT_ROOT / algorithm.value / f"{s}_seg.nii.gz").exists()]
        if done:
            print(f"Already segmented, skipping: {', '.join(done)}  (--force to redo)")
        subjects = [s for s in subjects if s not in done]

    if not subjects:
        print("Nothing to do.")
        return 0

    print(f"Algorithm : {algorithm.value}"
          f"{' (fallback)' if algorithm is not requested else ''}")
    print(f"Device    : {'CUDA GPU' if gpu else 'CPU only'}")
    print(f"Subjects  : {len(subjects)} -> {', '.join(subjects)}")
    print(f"Parallel  : {args.jobs}")
    print(f"Output    : {OUT_ROOT / algorithm.value}")
    if not gpu:
        per = 7 if algorithm is Algorithms.BraTS25_2 else 150
        eta = timedelta(minutes=per * len(subjects) / max(args.jobs, 1))
        print(f"Rough ETA : ~{eta} (upstream estimate, CPU, first run adds image pull)")
    print()

    if args.dry_run:
        return 0

    started = time.time()
    results: list[dict] = []

    if args.jobs == 1:
        for i, subject in enumerate(subjects, 1):
            print(f"[{i}/{len(subjects)}] {subject} ... ", end="", flush=True)
            r = segment_subject(subject, algorithm.value, force_cpu)
            results.append(r)
            print(f"{r['status']} ({r['seconds']}s)"
                  + (f"\n    {r['error']}" if r["status"] == "failed" else ""))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(segment_subject, s, algorithm.value, force_cpu): s
                       for s in subjects}
            for n, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                print(f"[{n}/{len(subjects)}] {r['subject']} {r['status']} "
                      f"({r['seconds']}s)"
                      + (f"\n    {r['error']}" if r["status"] == "failed" else ""))

    summary = {
        "algorithm": algorithm.value,
        "requested_algorithm": requested.value,
        "fell_back": algorithm is not requested,
        "device": "cuda" if gpu else "cpu",
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_seconds": round(time.time() - started, 1),
        "results": sorted(results, key=lambda r: r["subject"]),
    }
    report = OUT_ROOT / algorithm.value / "run_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2))

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{len(ok)}/{len(results)} succeeded in "
          f"{timedelta(seconds=int(summary['total_seconds']))}")
    print(f"Summary: {report}")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
