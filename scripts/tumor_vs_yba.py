#!/usr/bin/env python3
"""Intersect a tumour segmentation (in MNI space) with the Yale Brain Atlas.

Answers, per subject:
  * which YBA parcels the tumour overlaps, and by how much
  * which white matter tracts it crosses, and where those tracts terminate
  * a solid 3D surface for the tumour, for viewing against the atlas

Spaces
------
The YBA surface lives in MNI152 (x +/-72.6, y -107.6..73.7, z -52.4..82.5,
centroid ~(0,-17,16)). `register_to_mni.py` puts the tumour there. Both are then
in the same millimetre coordinate frame and can be compared directly.

How overlap is measured
-----------------------
The YBA is a *surface* atlas: ~63.8k labelled vertices, no volumetric label map
(see AtlasGeometry in the ElectrodeParcelGaussian project). So "which parcels
does this tumour hit" is answered by sampling the tumour label volume at each
atlas vertex's MNI coordinate, then aggregating per parcel.

That measures the share of each parcel's *surface* inside the tumour, which is
not the same as the share of its volume. For a cortical parcel, which is a
sheet, the two are close. Deep structures are represented by their boundary
surface, so a tumour sitting inside one without reaching its edge will be
under-counted. The vertex counts are reported so this stays visible.

Requires: run scripts/register_to_mni.py first.
Run with the YBAParcelVisualizer venv, which has the YaleBrainAtlas package:
    /home/vilem/Projects/YBAParcelVisualizer/.venv/bin/python scripts/tumor_vs_yba.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MNI_DIR = PROJECT_ROOT / "outputs" / "mni"
ATLAS_DIR = PROJECT_ROOT / "outputs" / "atlas"

LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}


def load_atlas():
    from YaleBrainAtlas import YaleBrainAtlas
    return YaleBrainAtlas("Subject")


def sample_at_points(seg_img, points_mm: np.ndarray) -> np.ndarray:
    """Look up the label volume at world (MNI mm) coordinates.

    Returns one label per point; points outside the volume get 0.
    """
    inv = np.linalg.inv(seg_img.affine)
    ijk = nib.affines.apply_affine(inv, points_mm)
    ijk = np.rint(ijk).astype(int)

    data = np.asarray(seg_img.dataobj)
    shape = np.array(data.shape)
    inside = np.all((ijk >= 0) & (ijk < shape), axis=1)

    out = np.zeros(len(points_mm), dtype=np.int16)
    sel = ijk[inside]
    out[inside] = data[sel[:, 0], sel[:, 1], sel[:, 2]]
    return out


def parcel_overlap(atlas, seg_img, min_vertices: int = 1) -> list[dict]:
    """Per-parcel tumour involvement, most involved first."""
    points = np.asarray(atlas.whole_brain.points, dtype=float)
    labels = np.asarray(atlas.whole_brain["parcel_labels"]).astype(str)
    hit = sample_at_points(seg_img, points)

    rows = []
    for parcel in np.unique(labels):
        mask = labels == parcel
        total = int(mask.sum())
        involved = hit[mask]
        n_hit = int((involved > 0).sum())
        if n_hit < min_vertices:
            continue
        breakdown = Counter(LABEL_NAMES.get(int(v), str(v))
                            for v in involved if v > 0)
        rows.append({
            "parcel": str(parcel),
            "vertices_total": total,
            "vertices_in_tumour": n_hit,
            "fraction": round(n_hit / total, 4),
            "dominant_tissue": breakdown.most_common(1)[0][0],
            "tissue_breakdown": dict(breakdown),
        })

    rows.sort(key=lambda r: (-r["fraction"], -r["vertices_in_tumour"]))
    return rows


def tract_overlap(atlas, seg_img, tract_names: list[str]) -> tuple[list[dict], list[str]]:
    """Per-tract tumour involvement plus terminal parcels.

    Returns (rows, skipped) -- skipped lists tracts whose geometry is not
    installed, which is the normal state unless the YBA WhiteMatterTracts data
    bundle is present.
    """
    rows, skipped = [], []
    for name in tract_names:
        try:
            tract = getattr(atlas, name)
            pts = np.asarray(tract.poly.points, dtype=float)
        except Exception:
            skipped.append(name)
            continue

        hit = sample_at_points(seg_img, pts)
        n_hit = int((hit > 0).sum())
        if n_hit == 0:
            continue

        try:
            ends = tract.get_start_and_end_parcels()
            start, end = list(ends.get("start", [])), list(ends.get("end", []))
        except Exception:
            start, end = [], []

        rows.append({
            "tract": name,
            "points_total": len(pts),
            "points_in_tumour": n_hit,
            "fraction": round(n_hit / len(pts), 4),
            "connects": {"start": start, "end": end},
        })

    rows.sort(key=lambda r: -r["fraction"])
    return rows, skipped


def tumour_mesh(seg_img, label_values=(1, 2, 3, 4)):
    """Marching-cubes surface of the tumour, with vertices in MNI mm.

    Built in voxel index space then pushed through the affine, so it lands in
    the same frame as the atlas regardless of the template's orientation.
    """
    import pyvista as pv

    data = np.isin(np.asarray(seg_img.dataobj), label_values).astype(np.float32)
    if data.sum() == 0:
        return None

    grid = pv.ImageData(dimensions=np.array(data.shape) + 1)
    grid.cell_data["v"] = data.flatten(order="F")
    surf = grid.cells_to_points("v").contour([0.5], scalars="v")
    if surf.n_points == 0:
        return None

    surf.points = nib.affines.apply_affine(seg_img.affine, surf.points)
    return surf.smooth(n_iter=30, relaxation_factor=0.1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", help="default: all registered subjects")
    ap.add_argument("--min-fraction", type=float, default=0.0,
                    help="only report parcels above this involved fraction")
    ap.add_argument("--top", type=int, default=25, help="rows to print")
    args = ap.parse_args()

    registered = sorted(
        p.name for p in MNI_DIR.iterdir()
        if p.is_dir() and (p / "seg_in_mni.nii.gz").exists()
    ) if MNI_DIR.is_dir() else []
    subjects = args.subjects or registered
    if not subjects:
        raise SystemExit(f"No registered subjects in {MNI_DIR}. "
                         f"Run scripts/register_to_mni.py first.")

    print("Loading Yale Brain Atlas ...", flush=True)
    atlas = load_atlas()
    tract_names = list(atlas.tract_names) if hasattr(atlas, "tract_names") else []
    print(f"  {len(atlas.parcel_names)} parcels, "
          f"{atlas.whole_brain.n_points} vertices, {len(tract_names)} tract names\n")

    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    for subject in subjects:
        seg_path = MNI_DIR / subject / "seg_in_mni.nii.gz"
        seg_img = nib.load(seg_path)

        parcels = parcel_overlap(atlas, seg_img)
        parcels = [p for p in parcels if p["fraction"] >= args.min_fraction]
        tracts, skipped = tract_overlap(atlas, seg_img, tract_names)

        mesh = tumour_mesh(seg_img)
        mesh_path = None
        if mesh is not None:
            mesh_path = ATLAS_DIR / f"{subject}_tumour.vtp"
            mesh.save(mesh_path)

        print(f"=== {subject} ===")
        print(f"{len(parcels)} parcel(s) involved"
              + (f", showing top {args.top}" if len(parcels) > args.top else ""))
        print(f"  {'parcel':<24}{'frac':>7}{'verts':>10}  tissue")
        for row in parcels[:args.top]:
            print(f"  {row['parcel']:<24}{row['fraction']:>7.1%}"
                  f"{row['vertices_in_tumour']:>5}/{row['vertices_total']:<4}"
                  f"  {row['dominant_tissue']}")

        if tracts:
            print(f"\n  {len(tracts)} tract(s) crossed:")
            for row in tracts[:args.top]:
                ends = row["connects"]
                print(f"  {row['tract']:<34}{row['fraction']:>7.1%}  "
                      f"{', '.join(ends['start'][:3])} <-> {', '.join(ends['end'][:3])}")
        elif skipped:
            print(f"\n  tracts: geometry unavailable for {len(skipped)} tract(s) "
                  f"- YBA WhiteMatterTracts data bundle not installed")

        if mesh_path:
            print(f"\n  tumour surface -> {mesh_path.relative_to(PROJECT_ROOT)} "
                  f"({mesh.n_points} pts, {mesh.volume/1000:.1f} mL enclosed)")
        print()

        report[subject] = {"parcels": parcels, "tracts": tracts,
                           "tracts_unavailable": skipped,
                           "mesh": str(mesh_path.relative_to(PROJECT_ROOT))
                                   if mesh_path else None}

    out = ATLAS_DIR / "yba_overlap.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
