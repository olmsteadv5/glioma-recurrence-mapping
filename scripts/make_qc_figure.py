#!/usr/bin/env python3
"""Render a QC overlay PNG per segmented subject: prediction on T1GD and FLAIR.

Picks the axial slice with the largest tumour cross-section, then shows
background / prediction / (reference, when published) side by side.

    outputs/<algorithm>/qc/<SUBJECT>_qc.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NII_DIR = PROJECT_ROOT / "data" / "PENN_GBM_nii"
SEG_DIR = PROJECT_ROOT / "data" / "reference_seg"
OUT_ROOT = PROJECT_ROOT / "outputs"

# BraTS 2025 labels -> colour
LABEL_COLORS = {1: "#f2c14e", 2: "#3fa7d6", 3: "#e4572e", 4: "#8e7dbe"}
LABEL_NAMES = {1: "NETC (necrosis)", 2: "SNFH (oedema)", 3: "ET (enhancing)",
               4: "RC (cavity)"}

_cmap = ListedColormap(["#00000000"] + [LABEL_COLORS[i] for i in (1, 2, 3, 4)])


def best_slice(seg: np.ndarray) -> int:
    """Axial slice holding the most tumour voxels (middle slice if no tumour)."""
    per_slice = (seg > 0).sum(axis=(0, 1))
    return int(per_slice.argmax()) if per_slice.max() else seg.shape[2] // 2


def show(ax, bg: np.ndarray, overlay: np.ndarray | None, z: int, title: str):
    ax.imshow(np.rot90(bg[:, :, z]), cmap="gray",
              vmin=0, vmax=np.percentile(bg[bg > 0], 99) if (bg > 0).any() else 1)
    if overlay is not None:
        masked = np.ma.masked_where(overlay[:, :, z] == 0, overlay[:, :, z])
        ax.imshow(np.rot90(masked), cmap=_cmap, vmin=0, vmax=4, alpha=0.5,
                  interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def render(subject: str, pred_path: Path, out_path: Path) -> None:
    subj_dir = NII_DIR / subject
    t1gd = np.asarray(nib.load(subj_dir / f"{subject}_T1GD.nii.gz").dataobj).astype(float)
    flair = np.asarray(nib.load(subj_dir / f"{subject}_FLAIR.nii.gz").dataobj).astype(float)
    pred = np.asarray(nib.load(pred_path).dataobj).astype(np.int16)

    ref_path = SEG_DIR / f"{subject}_seg.nii.gz"
    ref = (np.asarray(nib.load(ref_path).dataobj).astype(np.int16)
           if ref_path.exists() else None)

    z = best_slice(pred)
    panels = 3 if ref is None else 4
    fig, axes = plt.subplots(1, panels, figsize=(4 * panels, 4.6))

    show(axes[0], t1gd, None, z, f"T1GD (axial z={z})")
    show(axes[1], t1gd, pred, z, "Prediction on T1GD")
    show(axes[2], flair, pred, z, "Prediction on FLAIR")
    if ref is not None:
        show(axes[3], t1gd, ref, z, "Reference (expert)")

    present = sorted(set(np.unique(pred).astype(int).tolist()) - {0})
    fig.legend(handles=[mpatches.Patch(color=LABEL_COLORS[v], label=LABEL_NAMES[v])
                        for v in present if v in LABEL_COLORS],
               loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(subject, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


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

    made = 0
    for algo_dir in algo_dirs:
        for pred in sorted(algo_dir.glob("*_seg.nii.gz")):
            subject = pred.name.removesuffix("_seg.nii.gz")
            out = algo_dir / "qc" / f"{subject}_qc.png"
            render(subject, pred, out)
            print(f"wrote {out.relative_to(PROJECT_ROOT)}")
            made += 1
    if not made:
        print("No predictions to render yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
