#!/usr/bin/env python3
"""Summarise BraTS segmentations, and score them against reference labels.

Label convention (BraTS 2024/2025 adult glioma):
    0  background
    1  NETC  non-enhancing tumour core (necrosis)
    2  SNFH  surrounding non-enhancing FLAIR hyperintensity (oedema)
    3  ET    enhancing tissue
    4  RC    resection cavity          (post-treatment classes only)

Reported regions are the standard nested ones:
    ET  = {3}
    TC  = {1, 3}        tumour core
    WT  = {1, 2, 3}     whole tumour

The UPenn-GBM reference segmentations published with the cohort use the older
BraTS convention (1 = necrotic core, 2 = oedema, 4 = enhancing). They are
remapped onto the 2025 labels before scoring so the regions line up.

Usage:
    evaluate.py                       # summarise every prediction
    evaluate.py --algorithm BraTS25_2 # restrict to one algorithm's outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "outputs"
SEG_DIR = PROJECT_ROOT / "data" / "reference_seg"

LABEL_NAMES = {0: "background", 1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
REGIONS = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}

# UPenn-GBM / legacy BraTS labels -> BraTS 2025 labels
LEGACY_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    """Dice coefficient. None when the region is absent from both volumes."""
    total = a.sum() + b.sum()
    if total == 0:
        return None
    return float(2.0 * np.logical_and(a, b).sum() / total)


def to_2025_labels(data: np.ndarray) -> np.ndarray:
    """Remap a legacy-labelled reference volume if it looks legacy."""
    present = set(np.unique(data).astype(int).tolist())
    if 4 in present and 3 not in present:
        out = np.zeros_like(data)
        for src, dst in LEGACY_REMAP.items():
            out[data == src] = dst
        return out
    return data


def describe(path: Path) -> dict:
    """Voxel counts and volumes per label for one segmentation."""
    img = nib.load(path)
    data = np.asarray(img.dataobj).astype(np.int16)
    voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0

    labels = {}
    for value, name in LABEL_NAMES.items():
        if value == 0:
            continue
        count = int((data == value).sum())
        if count:
            labels[name] = {"voxels": count, "volume_ml": round(count * voxel_ml, 2)}

    regions = {}
    for name, values in REGIONS.items():
        count = int(np.isin(data, values).sum())
        regions[name] = {"voxels": count, "volume_ml": round(count * voxel_ml, 2)}

    return {"shape": list(img.shape), "labels": labels, "regions": regions,
            "unexpected_labels": sorted(set(np.unique(data).astype(int).tolist())
                                        - set(LABEL_NAMES))}


def score(pred_path: Path, ref_path: Path) -> dict:
    """Region-wise Dice between a prediction and a reference segmentation."""
    pred = np.asarray(nib.load(pred_path).dataobj).astype(np.int16)
    ref = to_2025_labels(np.asarray(nib.load(ref_path).dataobj).astype(np.int16))
    if pred.shape != ref.shape:
        return {"error": f"shape mismatch {pred.shape} vs {ref.shape}"}
    return {name: (None if (d := dice(np.isin(pred, v), np.isin(ref, v))) is None
                   else round(d, 4))
            for name, v in REGIONS.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", help="restrict to outputs/<algorithm>")
    args = ap.parse_args()

    algo_dirs = ([OUT_ROOT / args.algorithm] if args.algorithm
                 else sorted(p for p in OUT_ROOT.iterdir() if p.is_dir())
                 if OUT_ROOT.is_dir() else [])
    algo_dirs = [d for d in algo_dirs if d.is_dir()]
    if not algo_dirs:
        raise SystemExit(f"No segmentation outputs under {OUT_ROOT}")

    report: dict = {}
    for algo_dir in algo_dirs:
        preds = sorted(algo_dir.glob("*_seg.nii.gz"))
        if not preds:
            continue
        print(f"\n=== {algo_dir.name} ({len(preds)} subject(s)) ===")
        entries = {}
        for pred in preds:
            subject = pred.name.removesuffix("_seg.nii.gz")
            entry = describe(pred)

            ref = SEG_DIR / f"{subject}_seg.nii.gz"
            line = (f"{subject}  "
                    f"WT {entry['regions']['WT']['volume_ml']:7.1f} mL  "
                    f"TC {entry['regions']['TC']['volume_ml']:7.1f} mL  "
                    f"ET {entry['regions']['ET']['volume_ml']:7.1f} mL")
            if ref.exists():
                entry["dice_vs_reference"] = score(pred, ref)
                d = entry["dice_vs_reference"]
                if "error" not in d:
                    line += ("   Dice  "
                             + "  ".join(f"{k} {('  n/a' if d[k] is None else f'{d[k]:.3f}')}"
                                         for k in ("WT", "TC", "ET")))
                else:
                    line += f"   {d['error']}"
            if entry["unexpected_labels"]:
                line += f"   !! unexpected labels {entry['unexpected_labels']}"
            print(line)
            entries[subject] = entry
        report[algo_dir.name] = entries

    out = OUT_ROOT / "evaluation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
