#!/usr/bin/env python3
"""Fetch UPenn-GBM structural NIfTI scans and lay them out for the BraTS runner.

Source
------
The UPenn-GBM cohort (Bakas et al., Sci Data 2022) is published on TCIA. The
official route is the NBIA Data Retriever / Aspera client, which is awkward to
script. This uses a public GitHub-release mirror of the same collection that
exposes plain HTTPS assets:

    https://github.com/data-nih/tcia  (release tag: upenn-gbm)

The mirror renames subjects to BIDS-ish names. Verified mapping:

    sub-NNN          <->  UPENN-GBM-00NNN_11   (baseline / pre-op)
    sub-NNN_2        <->  UPENN-GBM-00NNN_21   (later timepoint)

    T1w.nii.gz       ->  T1     (native T1)
    ce-gd_T1w.nii.gz ->  T1GD   (post-contrast T1)
    T2w.nii.gz       ->  T2
    FLAIR.nii.gz     ->  FLAIR
    seg.nii.gz       ->  expert-approved reference segmentation (147 subjects)

This mirror is NOT an official TCIA endpoint. Files are checksummed and
sanity-checked on download (see verify_inputs.py), but for anything that leaves
the lab, re-pull from TCIA directly and cite the original collection.

Output layout matches the convention already in use for this project:

    data/PENN_GBM_nii/<SUBJECT>/<SUBJECT>_{T1,T1GD,T2,FLAIR}.nii.gz
    data/reference_seg/<SUBJECT>_seg.nii.gz      (when available)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = "data-nih/tcia"
TAG = "upenn-gbm"
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

# modality suffix in the mirror -> modality name used on disk
MODALITIES = {
    "T1w.nii.gz": "T1",
    "ce-gd_T1w.nii.gz": "T1GD",
    "T2w.nii.gz": "T2",
    "FLAIR.nii.gz": "FLAIR",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NII_DIR = DATA_DIR / "PENN_GBM_nii"
SEG_DIR = DATA_DIR / "reference_seg"


def subject_id(num: int, timepoint: int = 1) -> str:
    """Canonical UPenn-GBM subject id, e.g. 36 -> UPENN-GBM-00036_11."""
    return f"UPENN-GBM-{num:05d}_{timepoint}1"


def asset_name(num: int, suffix: str, timepoint: int = 1) -> str:
    """Mirror asset filename for a subject/modality/timepoint."""
    prefix = f"sub-{num:03d}" if timepoint == 1 else f"sub-{num:03d}_2"
    return f"{prefix}_{suffix}"


def download(url: str, dest: Path, force: bool = False) -> tuple[bool, str]:
    """Download url -> dest. Returns (downloaded, sha256). Skips existing files."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return False, sha256(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "brats-upenn-gbm/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"empty download: {url}")

    tmp.replace(dest)
    return True, sha256(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_subject(num: int, timepoint: int, with_seg: bool, force: bool) -> dict:
    """Fetch all 4 modalities (and optional reference seg) for one subject."""
    sid = subject_id(num, timepoint)
    record: dict = {"subject": sid, "sub_mirror": f"sub-{num:03d}", "files": {}}
    out_dir = NII_DIR / sid

    for suffix, modality in MODALITIES.items():
        name = asset_name(num, suffix, timepoint)
        dest = out_dir / f"{sid}_{modality}.nii.gz"
        try:
            fresh, digest = download(f"{BASE}/{name}", dest, force=force)
        except Exception as exc:
            print(f"  !! {sid} {modality}: {exc}", file=sys.stderr)
            record["files"][modality] = {"error": str(exc)}
            continue
        record["files"][modality] = {
            "path": str(dest.relative_to(PROJECT_ROOT)),
            "source_asset": name,
            "sha256": digest,
            "bytes": dest.stat().st_size,
        }
        print(f"  {'got ' if fresh else 'have'} {sid} {modality:5s} "
              f"({dest.stat().st_size / 1e6:.1f} MB)")

    # Reference segmentation only exists for baseline scans of 147 subjects.
    if with_seg and timepoint == 1:
        seg_dest = SEG_DIR / f"{sid}_seg.nii.gz"
        try:
            fresh, digest = download(f"{BASE}/sub-{num:03d}_seg.nii.gz", seg_dest,
                                     force=force)
            record["reference_seg"] = {
                "path": str(seg_dest.relative_to(PROJECT_ROOT)),
                "sha256": digest,
            }
            print(f"  {'got ' if fresh else 'have'} {sid} seg   (reference)")
        except Exception:
            print(f"  --   {sid} no reference segmentation published")

    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="*", type=int, default=[36, 42, 51],
                    help="UPenn-GBM subject numbers (default: 36 42 51)")
    ap.add_argument("--timepoint", type=int, default=1, choices=[1, 2],
                    help="1 = baseline/pre-op (_11), 2 = later scan (_21)")
    ap.add_argument("--with-seg", action="store_true",
                    help="also fetch the expert reference segmentation if published")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    print(f"Fetching {len(args.subjects)} subject(s) at timepoint {args.timepoint} "
          f"-> {NII_DIR}")

    records = []
    for num in args.subjects:
        print(f"[UPENN-GBM-{num:05d}]")
        records.append(fetch_subject(num, args.timepoint, args.with_seg, args.force))

    manifest = DATA_DIR / "manifest.json"
    existing = json.loads(manifest.read_text()) if manifest.exists() else []
    by_subject = {r["subject"]: r for r in existing}
    by_subject.update({r["subject"]: r for r in records})
    manifest.write_text(json.dumps(sorted(by_subject.values(),
                                          key=lambda r: r["subject"]), indent=2))
    print(f"\nManifest: {manifest}")

    failed = [r["subject"] for r in records
              if any("error" in v for v in r["files"].values())]
    if failed:
        print(f"FAILED (incomplete): {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
