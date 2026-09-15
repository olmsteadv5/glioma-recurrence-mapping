#!/usr/bin/env python3
"""Interactive 3D view: solid tumour inside the Yale Brain Atlas.

Renders, as one standalone HTML file per subject:
  * the tumour as a solid surface, split by tissue class
  * every YBA parcel it touches, opaque and individually named
  * the rest of the atlas as a faint shell (default 20% opacity) for context
  * crossed white matter tracts, when the YBA tract bundle is installed

Everything is in MNI152 millimetres, so the tumour and the atlas line up
directly -- see register_to_mni.py for how the tumour gets there.

Run with the YBAParcelVisualizer venv (it has the YaleBrainAtlas package):
    /home/vilem/Projects/YBAParcelVisualizer/.venv/bin/python \\
        scripts/visualize_yba.py
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MNI_DIR = PROJECT_ROOT / "outputs" / "mni"
ATLAS_DIR = PROJECT_ROOT / "outputs" / "atlas"

TISSUE = {1: ("NETC (necrosis)", "#f2c14e"),
          2: ("SNFH (oedema)", "#3fa7d6"),
          3: ("ET (enhancing)", "#e4572e"),
          4: ("RC (cavity)", "#8e7dbe")}


def distinct_colors(n: int) -> list[str]:
    """Evenly spaced hues, so adjacent parcels stay distinguishable."""
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 0.62, 0.95)
        out.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return out


def as_triangles(poly) -> np.ndarray:
    """pyvista face array -> (n, 3) triangle indices."""
    return np.asarray(poly.faces).reshape(-1, 4)[:, 1:]


def submesh(points: np.ndarray, tris: np.ndarray, keep: np.ndarray):
    """Faces whose vertices are all in `keep`, reindexed to a compact mesh."""
    face_mask = keep[tris].all(axis=1)
    if not face_mask.any():
        return None, None
    sel = tris[face_mask]
    used = np.unique(sel)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return points[used], remap[sel]


def tumour_traces(seg_path: Path, go):
    """One surface per tissue class, built by marching cubes in MNI space."""
    import pyvista as pv

    img = nib.load(seg_path)
    data = np.asarray(img.dataobj)
    traces = []

    for value, (label, color) in TISSUE.items():
        binary = (data == value).astype(np.float32)
        if binary.sum() < 10:
            continue
        grid = pv.ImageData(dimensions=np.array(binary.shape) + 1)
        grid.cell_data["v"] = binary.flatten(order="F")
        surf = grid.cells_to_points("v").contour([0.5], scalars="v")
        if surf.n_points == 0:
            continue
        surf = surf.smooth(n_iter=30, relaxation_factor=0.1)
        pts = nib.affines.apply_affine(img.affine, surf.points)
        tris = as_triangles(surf)
        traces.append(go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
            color=color, opacity=1.0, name=label, showlegend=True,
            hovertemplate=f"{label}<extra></extra>", flatshading=True))
    return traces


def build_figure(subject: str, atlas, overlap: dict, shell_opacity: float,
                 max_parcels: int):
    import plotly.graph_objects as go

    points = np.asarray(atlas.whole_brain.points, dtype=float)
    labels = np.asarray(atlas.whole_brain["parcel_labels"]).astype(str)
    tris = as_triangles(atlas.whole_brain)

    involved = [r["parcel"] for r in overlap.get("parcels", [])][:max_parcels]
    involved_set = set(involved)

    fig = go.Figure()

    # 1. Faint shell: everything not involved, one trace, cheap to render.
    rest = ~np.isin(labels, list(involved_set))
    pts, faces = submesh(points, tris, rest)
    if pts is not None:
        fig.add_trace(go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#9aa3ad", opacity=shell_opacity, name="atlas (uninvolved)",
            showlegend=True, hoverinfo="skip", flatshading=True))

    # 2. Involved parcels, one trace each so they can be toggled and named.
    palette = distinct_colors(len(involved))
    by_name = {r["parcel"]: r for r in overlap.get("parcels", [])}
    for color, parcel in zip(palette, involved):
        pts, faces = submesh(points, tris, labels == parcel)
        if pts is None:
            continue
        row = by_name[parcel]
        fig.add_trace(go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=0.95, name=f"{parcel} ({row['fraction']:.0%})",
            showlegend=True, flatshading=True,
            hovertemplate=(f"<b>{parcel}</b><br>"
                           f"{row['fraction']:.1%} of its volume in tumour<br>"
                           f"{row['volume_in_tumour_ml']:.2f} of "
                           f"{row['parcel_volume_ml']:.2f} mL<br>"
                           f"mostly {row['dominant_tissue']}<extra></extra>")))

    # 3. The tumour itself.
    for trace in tumour_traces(MNI_DIR / subject / "seg_in_mni.nii.gz", go):
        fig.add_trace(trace)

    # 4. Tracts, if the bundle is installed.
    for row in overlap.get("tracts", [])[:20]:
        try:
            tract = getattr(atlas, row["tract"])
            pts = np.asarray(tract.poly.points, dtype=float)
        except Exception:
            continue
        step = max(1, len(pts) // 4000)
        fig.add_trace(go.Scatter3d(
            x=pts[::step, 0], y=pts[::step, 1], z=pts[::step, 2],
            mode="markers", marker=dict(size=1.2, color="#2b2b2b"),
            name=f"{row['tract']} ({row['fraction']:.0%})", showlegend=True,
            hovertemplate=f"{row['tract']}<extra></extra>"))

    n_tracts = len(overlap.get("tracts", []))
    tot = overlap.get("tumour_volume_ml")
    inp = overlap.get("tumour_in_any_parcel_ml")
    vol = (f"{tot:.0f} mL tumour, {inp:.0f} mL ({inp / tot:.0%}) in parcelled "
           f"tissue. " if tot and inp is not None else "")
    subtitle = (vol + f"{len(involved)} parcels involved"
                + (f", {n_tracts} tracts crossed" if n_tracts
                   else " - tract data not installed"))
    fig.update_layout(
        title=f"{subject} - tumour in Yale Brain Atlas<br>"
              f"<sub>{subtitle}. MNI152 space. Click legend entries to toggle.</sub>",
        template="plotly_white",
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                   aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=70),
        legend=dict(itemsizing="constant", font=dict(size=10)))
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--shell-opacity", type=float, default=0.2,
                    help="opacity of uninvolved atlas parcels (default 0.2)")
    ap.add_argument("--max-parcels", type=int, default=30,
                    help="how many involved parcels to draw individually")
    args = ap.parse_args()

    overlap_path = ATLAS_DIR / "yba_overlap.json"
    if not overlap_path.exists():
        raise SystemExit(f"{overlap_path} not found - run scripts/tumor_vs_yba.py first")
    report = json.loads(overlap_path.read_text())

    subjects = args.subjects or sorted(report)
    print("Loading Yale Brain Atlas ...", flush=True)
    from YaleBrainAtlas import YaleBrainAtlas
    atlas = YaleBrainAtlas("Subject")

    for subject in subjects:
        if subject not in report:
            print(f"skip {subject}: not in overlap report")
            continue
        print(f"rendering {subject} ...", end="", flush=True)
        fig = build_figure(subject, atlas, report[subject],
                           args.shell_opacity, args.max_parcels)
        out = ATLAS_DIR / f"{subject}_yba.html"
        fig.write_html(out, include_plotlyjs="cdn")
        print(f" -> {out.relative_to(PROJECT_ROOT)}")

    print("\nOpen the HTML in a browser: rotate with drag, zoom with scroll,")
    print("click legend entries to show/hide individual parcels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
