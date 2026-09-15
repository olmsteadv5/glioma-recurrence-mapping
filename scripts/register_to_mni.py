#!/usr/bin/env python3
"""Register a UPenn-GBM subject from SRI24 space into MNI152, and carry the
tumour segmentation along with it.

Why this is needed
------------------
BraTS outputs sit on the SRI24 atlas grid (240x240x155, 1mm, LPS). The Yale
Brain Atlas is defined in MNI152 space -- its surface spans x +/-72.6,
y -107.6..73.7, z -52.4..82.5 with centroid (0, -17, 16), which matches the
MNI152 1mm template. To ask "which YBA parcels does this tumour hit", the
tumour has to be moved into MNI first.

Cost-function masking (two passes)
---------------------------------
A glioma displaces healthy tissue (mass effect). A deformable registration
driven by image similarity will happily squash or stretch the tumour to make
the patient look more like the template, which corrupts the warp *around* the
lesion -- exactly the region we care about.

The fix is to exclude the lesion from the similarity metric. greedy applies
masks to the *fixed* image (`-gm`); its moving-image mask (`-mm`) errors out in
the ITK-SNAP build here. Since the fixed image is the MNI template, the mask has
to be defined in MNI space -- but the tumour starts in patient space, so we do
not have it there yet. Hence two passes:

  pass 1  register unmasked, and push the tumour into MNI to find roughly
          where the lesion lands in template space
  pass 2  build an MNI-space mask excluding that region (dilated), and
          re-register with `-gm` so the lesion drives nothing

Masking the reference-space lesion region is the same convention FSL uses for
cost-function masking. It is a mitigation, not a cure: tissue displaced by a
large lesion is still displaced. Treat parcel assignments near the tumour rim
as approximate.

Outputs (per subject, under outputs/mni/<SUBJECT>/):
    affine.mat            SRI24 -> MNI affine (pass 2)
    warp.nii.gz           deformable field (pass 2)
    fixed_mask.nii.gz     MNI-space metric mask (lesion region excluded)
    t1_in_mni.nii.gz      patient T1 resampled into MNI (check this visually!)
    seg_in_mni.nii.gz     tumour labels in MNI == YBA space
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
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NII_DIR = PROJECT_ROOT / "data" / "PENN_GBM_nii"
OUT_ROOT = PROJECT_ROOT / "outputs"
MNI_DEFAULT = Path("/home/vilem/Documents/SeizureAnalysis/from_downloads/"
                   "MNI152_T1_1mm_Brain.nii.gz")


def run(cmd: list[str], log: Path) -> None:
    """Run a command, appending stdout+stderr to log. Raise on failure."""
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        fh.flush()
        proc = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
                           f"see {log}")


def _brain_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj) > 0


def _save_like(data: np.ndarray, ref: Path, out: Path) -> None:
    ref_img = nib.load(ref)
    img = nib.Nifti1Image(data.astype(np.uint8), ref_img.affine)
    img.set_data_dtype(np.uint8)
    nib.save(img, out)


def _affine(fixed: Path, moving: Path, out: Path, log: Path, threads: str,
            mask: Path | None) -> None:
    cmd = ["greedy", "-d", "3", "-a", "-m", "NMI", "-i", fixed, moving]
    if mask:
        cmd += ["-gm", mask]
    run(cmd + ["-o", out, "-ia-image-centers", "-n", "100x100x50",
               "-threads", threads], log)


def _deformable(fixed: Path, moving: Path, init: Path, out: Path, log: Path,
                threads: str, mask: Path | None) -> None:
    cmd = ["greedy", "-d", "3", "-m", "NCC", "2x2x2", "-i", fixed, moving]
    if mask:
        cmd += ["-gm", mask]
    run(cmd + ["-it", init, "-o", out, "-n", "100x50x20",
               "-threads", threads], log)


def _reslice_seg(mni: Path, seg: Path, out: Path, warp: Path, affine: Path,
                 log: Path, threads: str) -> None:
    """Resample the label map into MNI using LABEL interpolation.

    LABEL smooths each label separately and takes the argmax; plain
    nearest-neighbour aliases badly through a deformable warp.
    """
    run(["greedy", "-d", "3", "-rf", mni,
         "-ri", "LABEL", "0.2vox", "-rm", seg, out,
         "-r", warp, affine, "-threads", threads], log)


def register(subject: str, algorithm: str, mni: Path, threads: str,
             force: bool, dilate_mm: int = 5) -> dict:
    subj_dir = NII_DIR / subject
    t1 = subj_dir / f"{subject}_T1.nii.gz"
    seg = OUT_ROOT / algorithm / f"{subject}_seg.nii.gz"

    for p in (t1, seg, mni):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    out_dir = OUT_ROOT / "mni" / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "registration.log"

    affine = out_dir / "affine.mat"
    warp = out_dir / "warp.nii.gz"
    t1_mni = out_dir / "t1_in_mni.nii.gz"
    seg_mni = out_dir / "seg_in_mni.nii.gz"
    fixed_mask = out_dir / "fixed_mask.nii.gz"
    tmp_affine = out_dir / "_pass1_affine.mat"
    tmp_warp = out_dir / "_pass1_warp.nii.gz"
    tmp_seg = out_dir / "_pass1_seg.nii.gz"

    if seg_mni.exists() and not force:
        print(f"  have {subject} (--force to redo)")
        return {"subject": subject, "status": "cached",
                "seg_in_mni": str(seg_mni.relative_to(PROJECT_ROOT))}

    log.write_text("")
    started = time.time()

    # --- pass 1: unmasked, only to locate the lesion in template space -------
    print("  pass 1 (locate lesion in MNI) ...", end="", flush=True)
    _affine(mni, t1, tmp_affine, log, threads, None)
    _deformable(mni, t1, tmp_affine, tmp_warp, log, threads, None)
    _reslice_seg(mni, seg, tmp_seg, tmp_warp, tmp_affine, log, threads)
    print(" done")

    # --- build the MNI-space metric mask -------------------------------------
    lesion = np.asarray(nib.load(tmp_seg).dataobj) > 0
    if dilate_mm > 0:
        # Dilate so the peritumoural rim, where displacement is worst, is also
        # kept out of the metric.
        lesion = ndimage.binary_dilation(lesion, iterations=dilate_mm)
    template_brain = _brain_mask(mni)
    mask = template_brain & ~lesion
    _save_like(mask, mni, fixed_mask)

    kept = float(mask.sum()) / max(int(template_brain.sum()), 1)
    print(f"  metric mask keeps {kept:.1%} of template brain "
          f"({'ok' if kept > 0.6 else 'LOW - large lesion, warp is on thin evidence'})")

    # --- pass 2: masked ------------------------------------------------------
    print("  pass 2 (masked registration) ...", end="", flush=True)
    _affine(mni, t1, affine, log, threads, fixed_mask)
    _deformable(mni, t1, affine, warp, log, threads, fixed_mask)
    print(" done")

    print("  reslicing ...", end="", flush=True)
    run(["greedy", "-d", "3", "-rf", mni, "-rm", t1, t1_mni,
         "-r", warp, affine, "-threads", threads], log)
    _reslice_seg(mni, seg, seg_mni, warp, affine, log, threads)
    print(" done")

    for tmp in (tmp_affine, tmp_warp, tmp_seg):
        tmp.unlink(missing_ok=True)

    before = np.asarray(nib.load(seg).dataobj)
    after = np.asarray(nib.load(seg_mni).dataobj)
    lost = sorted(set(np.unique(before).tolist()) - set(np.unique(after).tolist()))

    return {
        "subject": subject,
        "status": "ok",
        "seconds": round(time.time() - started, 1),
        "template_brain_kept": round(kept, 3),
        "volume_before_ml": round(float((before > 0).sum()) / 1000, 1),
        "volume_after_ml": round(float((after > 0).sum()) / 1000, 1),
        "labels_lost": lost,
        "seg_in_mni": str(seg_mni.relative_to(PROJECT_ROOT)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", help="default: all segmented subjects")
    ap.add_argument("--algorithm", default="BraTS25_2")
    ap.add_argument("--mni", type=Path, default=MNI_DEFAULT,
                    help="MNI152 1mm skull-stripped template (the YBA reference space)")
    ap.add_argument("--threads", default="8")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if shutil.which("greedy") is None:
        raise SystemExit("greedy not found (ships with ITK-SNAP)")

    seg_dir = OUT_ROOT / args.algorithm
    subjects = args.subjects or sorted(
        p.name.removesuffix("_seg.nii.gz") for p in seg_dir.glob("*_seg.nii.gz"))
    if not subjects:
        raise SystemExit(f"no segmentations in {seg_dir}")

    print(f"Registering {len(subjects)} subject(s) SRI24 -> MNI152 (YBA space)")
    print(f"Template: {args.mni}\n")

    results = []
    for i, subject in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] {subject}")
        try:
            r = register(subject, args.algorithm, args.mni, args.threads, args.force)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            r = {"subject": subject, "status": "failed", "error": str(exc)}
        results.append(r)
        if r.get("status") == "ok":
            print(f"  {r['seconds']}s | tumour {r['volume_before_ml']} -> "
                  f"{r['volume_after_ml']} mL"
                  + (f" | LOST LABELS {r['labels_lost']}" if r["labels_lost"] else ""))

    ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
    print(f"\n{ok}/{len(results)} registered")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
