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
Every YBA parcel is a *closed* mesh -- each of the 696 has zero open boundary
edges, so it encloses a real volume (median 0.80 mL, max 3.75 mL). Overlap is
therefore a genuine volume intersection, in mL, not a surface-proximity proxy.

The parcel meshes are voxelised once onto the MNI grid (point-in-mesh test per
parcel over its own bounding box) and cached as a label volume. After that,
intersecting a tumour with the atlas is voxel counting.

Two things to keep in mind when reading the numbers:

  * The 696 parcels total ~642 mL against a ~1200-1500 mL brain, i.e. they
    cover roughly half of it. Deep white matter is not parcellated. A tumour
    can be large and still overlap little, which is correct, not a failure --
    `tumour_in_any_parcel_ml` reports how much of it landed in parcelled
    tissue at all.
  * Voxelisation is on a 1 mm grid, so parcels near the small end (0.08 mL)
    are only a handful of voxels and their percentages are coarse.

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
    """Look up a label volume at world (MNI mm) coordinates.

    Returns one label per point; points outside the volume get 0.
    """
    inv = np.linalg.inv(seg_img.affine)
    ijk = np.rint(nib.affines.apply_affine(inv, points_mm)).astype(int)

    data = np.asarray(seg_img.dataobj)
    shape = np.array(data.shape)
    inside = np.all((ijk >= 0) & (ijk < shape), axis=1)

    out = np.zeros(len(points_mm), dtype=np.int16)
    sel = ijk[inside]
    out[inside] = data[sel[:, 0], sel[:, 1], sel[:, 2]]
    return out


def parcel_meshes(atlas):
    """Every parcel as a closed trimesh, in MNI millimetres.

    Parcels are extracted as the faces whose three vertices all carry the same
    label, then reindexed. Each comes out watertight.
    """
    import trimesh

    points = np.asarray(atlas.whole_brain.points, dtype=float)
    labels = np.asarray(atlas.whole_brain["parcel_labels"]).astype(str)
    tris = np.asarray(atlas.whole_brain.faces).reshape(-1, 4)[:, 1:]

    out = {}
    for name in np.unique(labels):
        keep = labels == name
        sel = tris[keep[tris].all(axis=1)]
        if len(sel) == 0:
            continue
        used = np.unique(sel)
        remap = np.full(len(points), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        out[str(name)] = trimesh.Trimesh(vertices=points[used],
                                         faces=remap[sel], process=False)
    return out


def build_label_volume(atlas, ref_img, cache: Path):
    """Voxelise all parcels onto ref_img's grid. Cached -- this is the slow bit.

    Returns (label array of parcel indices, names list). Index 0 means "no
    parcel", which covers deep white matter and everything else the atlas does
    not parcellate.
    """
    names_path = cache.with_suffix(".names.json")
    if cache.exists() and names_path.exists():
        return (np.asarray(nib.load(cache).dataobj).astype(np.int32),
                json.loads(names_path.read_text()))

    print("  voxelising 696 parcels onto the MNI grid (one-off, cached) ...",
          end="", flush=True)
    shape = ref_img.shape
    labels = np.zeros(shape, dtype=np.int32)
    meshes = parcel_meshes(atlas)
    names = sorted(meshes)
    inv = np.linalg.inv(ref_img.affine)

    for idx, name in enumerate(names, start=1):
        mesh = meshes[name]
        # Work only inside this parcel's bounding box; they are ~1 mL each, so
        # this keeps the point-in-mesh test to a few thousand voxels.
        corners = np.array([[x, y, z]
                            for x in mesh.bounds[:, 0]
                            for y in mesh.bounds[:, 1]
                            for z in mesh.bounds[:, 2]])
        vox = nib.affines.apply_affine(inv, corners)
        lo = np.floor(vox.min(axis=0)).astype(int) - 1
        hi = np.ceil(vox.max(axis=0)).astype(int) + 2
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, shape)
        if np.any(hi <= lo):
            continue

        grid = np.stack(np.meshgrid(
            np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]),
            np.arange(lo[2], hi[2]), indexing="ij"), axis=-1).reshape(-1, 3)
        world = nib.affines.apply_affine(ref_img.affine, grid)
        inside = mesh.contains(world)
        if inside.any():
            sel = grid[inside]
            # First parcel to claim a voxel keeps it; parcels barely overlap.
            free = labels[sel[:, 0], sel[:, 1], sel[:, 2]] == 0
            sel = sel[free]
            labels[sel[:, 0], sel[:, 1], sel[:, 2]] = idx

    cache.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(labels.astype(np.int16), ref_img.affine), cache)
    names_path.write_text(json.dumps(names))
    print(f" done ({(labels > 0).sum() / 1000:.0f} mL parcelled)")
    return labels, names


def parcel_overlap(labels: np.ndarray, names: list[str], seg_img) -> list[dict]:
    """Volume intersection, in mL, between the tumour and every parcel."""
    tumour = np.asarray(seg_img.dataobj).astype(np.int16)
    if tumour.shape != labels.shape:
        raise SystemExit(f"grid mismatch {tumour.shape} vs {labels.shape}")

    voxel_ml = float(np.prod(seg_img.header.get_zooms()[:3])) / 1000.0
    hit = tumour > 0

    # One pass each: total voxels per parcel, and tumour voxels per parcel.
    n = len(names) + 1
    total = np.bincount(labels.ravel(), minlength=n)
    inside = np.bincount(labels[hit].ravel(), minlength=n)

    rows = []
    for idx, name in enumerate(names, start=1):
        if inside[idx] == 0:
            continue
        sub = tumour[hit & (labels == idx)]
        breakdown = {LABEL_NAMES.get(int(v), str(v)): int((sub == v).sum())
                     for v in np.unique(sub)}
        rows.append({
            "parcel": name,
            "parcel_volume_ml": round(total[idx] * voxel_ml, 3),
            "volume_in_tumour_ml": round(inside[idx] * voxel_ml, 3),
            "fraction": round(float(inside[idx]) / total[idx], 4),
            "dominant_tissue": max(breakdown, key=breakdown.get),
            "tissue_breakdown": {k: round(v * voxel_ml, 3)
                                 for k, v in breakdown.items()},
        })

    rows.sort(key=lambda r: (-r["fraction"], -r["volume_in_tumour_ml"]))
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

    # The parcel label volume is built once against the first subject's grid;
    # every subject is resliced onto the same MNI template, so it is reusable.
    first = nib.load(MNI_DIR / subjects[0] / "seg_in_mni.nii.gz")
    labels, names = build_label_volume(atlas, first, ATLAS_DIR / "yba_labels_mni.nii.gz")
    print()

    for subject in subjects:
        seg_path = MNI_DIR / subject / "seg_in_mni.nii.gz"
        seg_img = nib.load(seg_path)

        parcels = parcel_overlap(labels, names, seg_img)
        parcels = [p for p in parcels if p["fraction"] >= args.min_fraction]
        tracts, skipped = tract_overlap(atlas, seg_img, tract_names)

        mesh = tumour_mesh(seg_img)
        mesh_path = None
        if mesh is not None:
            mesh_path = ATLAS_DIR / f"{subject}_tumour.vtp"
            mesh.save(mesh_path)

        voxel_ml = float(np.prod(seg_img.header.get_zooms()[:3])) / 1000.0
        tumour_ml = float((np.asarray(seg_img.dataobj) > 0).sum()) * voxel_ml
        in_parcels = sum(r["volume_in_tumour_ml"] for r in parcels)

        print(f"=== {subject} ===")
        print(f"tumour {tumour_ml:.1f} mL, of which {in_parcels:.1f} mL "
              f"({in_parcels / max(tumour_ml, 1e-9):.0%}) lies in parcelled tissue")
        print(f"{len(parcels)} parcel(s) involved"
              + (f", showing top {args.top}" if len(parcels) > args.top else ""))
        print(f"  {'parcel':<24}{'frac':>7}{'in tumour':>12}{'of':>9}  tissue")
        for row in parcels[:args.top]:
            print(f"  {row['parcel']:<24}{row['fraction']:>7.1%}"
                  f"{row['volume_in_tumour_ml']:>10.2f} mL"
                  f"{row['parcel_volume_ml']:>7.2f}  {row['dominant_tissue']}")

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

        report[subject] = {"tumour_volume_ml": round(tumour_ml, 2),
                           "tumour_in_any_parcel_ml": round(in_parcels, 2),
                           "parcels": parcels, "tracts": tracts,
                           "tracts_unavailable": skipped,
                           "mesh": str(mesh_path.relative_to(PROJECT_ROOT))
                                   if mesh_path else None}

    out = ATLAS_DIR / "yba_overlap.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
