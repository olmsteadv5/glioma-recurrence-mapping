#!/usr/bin/env python3
"""Skull-strip the UPenn-GBM scans.

Why this exists
---------------
TCIA publishes UPenn-GBM in two forms: `images_structural` (skull-stripped) and
`images_structural_unstripped`. The GitHub mirror this project downloads from
ships the **unstripped** one -- a T1 comes in around 2900 mL of foreground,
where a stripped brain is 1200-1600 mL.

That breaks two things downstream:
  * BraTS algorithms are trained on skull-stripped input and assume it
  * registration to a skull-stripped MNI template is partly driven by skull and
    scalp matching against nothing

One mask for all four modalities
--------------------------------
The mask is computed once on the T1 and applied to T1/T1GD/T2/FLAIR. The
modalities are already co-registered, so one mask is valid for all of them, and
running SynthStrip per modality would give four slightly different brain
outlines -- which would then disagree about which voxels exist.

The originals are moved to data/PENN_GBM_unstripped/ rather than deleted, so the
raw download is still there and this stays reversible.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NII_DIR = PROJECT_ROOT / "data" / "PENN_GBM_nii"
RAW_DIR = PROJECT_ROOT / "data" / "PENN_GBM_unstripped"
MASK_DIR = PROJECT_ROOT / "data" / "brain_masks"

MODALITIES = ("T1", "T1GD", "T2", "FLAIR")
SYNTHSTRIP = Path("/opt/freesurfer/bin/mri_synthstrip")

# A stripped adult brain, tumour and oedema included, sits well under this.
STRIPPED_MAX_ML = 2000.0


def volume_ml(path: Path) -> float:
    return float((np.asarray(nib.load(path).dataobj) > 0).sum()) / 1000.0


def looks_stripped(subj_dir: Path, subject: str) -> bool:
    t1 = subj_dir / f"{subject}_T1.nii.gz"
    return t1.exists() and volume_ml(t1) < STRIPPED_MAX_ML


def strip_subject(subject: str, threads: str, force: bool) -> dict:
    subj_dir = NII_DIR / subject
    t1 = subj_dir / f"{subject}_T1.nii.gz"
    if not t1.exists():
        return {"subject": subject, "status": "failed", "error": "no T1"}

    if looks_stripped(subj_dir, subject) and not force:
        return {"subject": subject, "status": "already_stripped",
                "volume_ml": round(volume_ml(t1), 1)}

    started = time.time()
    before = volume_ml(t1)

    # Back up the originals once, so a re-run never strips a stripped image.
    raw_dir = RAW_DIR / subject
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        for m in MODALITIES:
            src = subj_dir / f"{subject}_{m}.nii.gz"
            if src.exists():
                shutil.copy2(src, raw_dir / src.name)

    # 1. One mask, from the T1.
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = MASK_DIR / f"{subject}_brainmask.nii.gz"
    proc = subprocess.run(
        [str(SYNTHSTRIP), "-i", str(raw_dir / t1.name), "-m", str(mask_path),
         "-t", str(threads)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return {"subject": subject, "status": "failed",
                "error": f"synthstrip: {proc.stderr.strip()[:200]}"}

    mask = np.asarray(nib.load(mask_path).dataobj) > 0

    # 2. Apply it to every modality.
    for m in MODALITIES:
        src = raw_dir / f"{subject}_{m}.nii.gz"
        if not src.exists():
            continue
        img = nib.load(src)
        data = np.asarray(img.dataobj)
        out = np.where(mask, data, 0).astype(img.get_data_dtype())
        new = nib.Nifti1Image(out, img.affine, img.header)
        new.set_data_dtype(img.get_data_dtype())
        nib.save(new, subj_dir / f"{subject}_{m}.nii.gz")

    after = volume_ml(subj_dir / t1.name)
    return {"subject": subject, "status": "ok",
            "seconds": round(time.time() - started, 1),
            "volume_before_ml": round(before, 1),
            "volume_after_ml": round(after, 1),
            "removed_pct": round(100 * (1 - after / before), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", help="default: all subjects in data/")
    ap.add_argument("--threads", default="8")
    ap.add_argument("--force", action="store_true",
                    help="re-strip from the backed-up originals")
    args = ap.parse_args()

    if not SYNTHSTRIP.exists():
        raise SystemExit(f"mri_synthstrip not found at {SYNTHSTRIP}")
    if not NII_DIR.is_dir():
        raise SystemExit(f"no input directory: {NII_DIR}")

    subjects = args.subjects or sorted(p.name for p in NII_DIR.iterdir() if p.is_dir())
    print(f"Skull-stripping {len(subjects)} subject(s) with SynthStrip\n")

    results = []
    for i, subject in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] {subject} ... ", end="", flush=True)
        r = strip_subject(subject, args.threads, args.force)
        results.append(r)
        if r["status"] == "ok":
            print(f"{r['volume_before_ml']} -> {r['volume_after_ml']} mL "
                  f"(-{r['removed_pct']}%) in {r['seconds']}s")
        elif r["status"] == "already_stripped":
            print(f"already stripped ({r['volume_ml']} mL)")
        else:
            print(f"FAILED: {r['error']}", file=sys.stderr)

    ok = sum(1 for r in results if r["status"] in ("ok", "already_stripped"))
    print(f"\n{ok}/{len(results)} ready. Originals kept in "
          f"{RAW_DIR.relative_to(PROJECT_ROOT)}/")
    if ok:
        print("Re-run scripts/run_segmentation.py --force and "
              "scripts/register_to_mni.py --force on the stripped images.")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
