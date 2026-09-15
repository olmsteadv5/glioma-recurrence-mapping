#!/usr/bin/env python3
"""Sanity-check downloaded UPenn-GBM volumes before spending CPU hours on them.

BraTS algorithms assume their inputs are already co-registered, skull-stripped
and resampled to the SRI24 atlas grid. They do not check, and they do not fail
loudly when the assumption is broken — they just return a poor segmentation.
This checks the assumption explicitly.

Checks per subject:
  * all four modalities present
  * 240x240x155 at 1 mm isotropic (the BraTS/SRI24 grid)
  * identical affine across modalities (i.e. actually co-registered)
  * plausible skull-stripped foreground fraction
  * non-empty, non-constant intensities
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NII_DIR = PROJECT_ROOT / "data" / "PENN_GBM_nii"

MODALITIES = ("T1", "T1GD", "T2", "FLAIR")
EXPECTED_SHAPE = (240, 240, 155)
EXPECTED_ZOOMS = (1.0, 1.0, 1.0)
# Absolute brain volume is the check that actually separates a stripped brain
# from a whole head. A fraction-of-grid test does not: an unstripped UPenn-GBM
# head is ~33% of the 240x240x155 grid, which sits inside any plausible band for
# a stripped brain. Adult brain is ~1100-1600 mL; add oedema, mass effect and a
# generous margin and anything over 2000 mL still has skull attached.
BRAIN_VOLUME_ML = (900.0, 2000.0)
# Kept as a loose floor for "is there an image here at all".
FOREGROUND_RANGE = (0.05, 0.60)


def check_subject(subj_dir: Path) -> list[str]:
    """Return a list of problems; empty means the subject looks usable."""
    problems: list[str] = []
    name = subj_dir.name
    affines = {}

    for modality in MODALITIES:
        path = subj_dir / f"{name}_{modality}.nii.gz"
        if not path.exists():
            problems.append(f"missing {modality}")
            continue

        try:
            img = nib.load(path)
            data = np.asarray(img.dataobj)
        except Exception as exc:
            problems.append(f"{modality}: unreadable ({exc})")
            continue

        if img.shape != EXPECTED_SHAPE:
            problems.append(f"{modality}: shape {img.shape} != {EXPECTED_SHAPE}")

        zooms = tuple(round(float(z), 3) for z in img.header.get_zooms()[:3])
        if zooms != EXPECTED_ZOOMS:
            problems.append(f"{modality}: voxel size {zooms} != {EXPECTED_ZOOMS}")

        affines[modality] = img.affine

        finite = np.isfinite(data)
        if not finite.all():
            problems.append(f"{modality}: contains NaN/Inf")
        if data.max() == data.min():
            problems.append(f"{modality}: constant image (all {data.min()})")
            continue

        fg = float((data > 0).sum()) / data.size
        if not FOREGROUND_RANGE[0] <= fg <= FOREGROUND_RANGE[1]:
            problems.append(
                f"{modality}: foreground {fg:.1%} outside "
                f"{FOREGROUND_RANGE[0]:.0%}-{FOREGROUND_RANGE[1]:.0%} "
                f"(empty or corrupt?)")

        voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
        brain_ml = float((data > 0).sum()) * voxel_ml
        if brain_ml > BRAIN_VOLUME_ML[1]:
            problems.append(
                f"{modality}: {brain_ml:.0f} mL of foreground, over "
                f"{BRAIN_VOLUME_ML[1]:.0f} mL - NOT skull-stripped. "
                f"Run scripts/skullstrip.py")
        elif brain_ml < BRAIN_VOLUME_ML[0]:
            problems.append(
                f"{modality}: only {brain_ml:.0f} mL of foreground, under "
                f"{BRAIN_VOLUME_ML[0]:.0f} mL - over-stripped or truncated?")

    # Co-registration: every modality must share one affine.
    if len(affines) > 1:
        ref_name, ref = next(iter(affines.items()))
        for modality, aff in affines.items():
            if modality != ref_name and not np.allclose(aff, ref, atol=1e-4):
                problems.append(f"{modality}: affine differs from {ref_name} "
                                f"(not co-registered?)")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", help="default: all subjects in data/")
    args = ap.parse_args()

    if not NII_DIR.is_dir():
        raise SystemExit(f"No input directory: {NII_DIR}")

    dirs = sorted(p for p in NII_DIR.iterdir() if p.is_dir())
    if args.subjects:
        wanted = set(args.subjects)
        dirs = [d for d in dirs if d.name in wanted or d.name.split("_")[0] in wanted]
    if not dirs:
        raise SystemExit("No subjects found.")

    bad = 0
    for subj_dir in dirs:
        problems = check_subject(subj_dir)
        if problems:
            bad += 1
            print(f"FAIL {subj_dir.name}")
            for p in problems:
                print(f"       {p}")
        else:
            t1 = subj_dir / f"{subj_dir.name}_T1.nii.gz"
            img = nib.load(t1)
            ml = float((np.asarray(img.dataobj) > 0).sum()) * (
                float(np.prod(img.header.get_zooms()[:3])) / 1000.0)
            print(f"ok   {subj_dir.name}  "
                  f"{'/'.join(MODALITIES)} @ {EXPECTED_SHAPE}, 1mm, "
                  f"co-registered, {ml:.0f} mL brain")

    print(f"\n{len(dirs) - bad}/{len(dirs)} subject(s) ready for segmentation")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
