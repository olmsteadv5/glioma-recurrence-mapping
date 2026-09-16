#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Interactive 3D view: solid tumour inside the Yale Brain Atlas.

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import pyvista as pv

import tracts as tractography
from atlas import brain, colours, label_volume, meshes, parcels
from paths import LABELS, MNI, out_dir

TISSUE_COLOR = {1: "#f2c14e", 2: "#3fa7d6", 3: "#e4572e", 4: "#8e7dbe"}
SHELL_COLOR = "#9aa3ad"
TRACT_COLOR = "#2f6f4e"


def _tumour_surface(data, affine, value):
    # Surface around one tissue class.
    binary = (data == value).astype(np.float32)
    if binary.sum() < 10:
        return None
    grid = pv.ImageData(dimensions=np.array(binary.shape) + 1)
    grid.cell_data["v"] = binary.flatten(order="F")
    surface = grid.cells_to_points("v").contour([0.5], scalars="v")
    if surface.n_points == 0:
        return None
    surface = surface.smooth(n_iter=30, relaxation_factor=0.1)
    return (nib.affines.apply_affine(affine, surface.points),
            np.asarray(surface.faces).reshape(-1, 4)[:, 1:])


def _mesh3d(vertices, faces, **kwargs):
    return go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                     flatshading=True, **kwargs)


def add(subject, fig, opacity=1.0, rank=10):
    # Add the tumour to a figure that already exists.
    image = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")
    data = np.asarray(image.dataobj).astype(np.int16)
    for value, name in LABELS.items():
        surface = _tumour_surface(data, image.affine, value)
        if surface is None:
            continue
        fig.add_trace(_mesh3d(*surface, color=TISSUE_COLOR[value], opacity=opacity,
                              name=name, showlegend=True, legendrank=rank + value,
                              hovertemplate=f"{name}<extra></extra>"))
    return fig


def add_tracts(subject, overlap, fig, width=1.5):
    # Tract lines through the tumour and the parcels around it.
    labels, names = label_volume(nib.load(MNI))
    index = {name: i + 1 for i, name in enumerate(names)}
    involved = [index[r["parcel"]] for r in overlap["parcels"] if r["parcel"] in index]
    seg = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")

    lines = tractography.crossing(tractography.centrelines(), seg, labels, involved)
    if not lines:
        return fig

    # all in one trace, None separates the lines
    xs, ys, zs = [], [], []
    for line in lines:
        xs += [*line[:, 0], None]
        ys += [*line[:, 1], None]
        zs += [*line[:, 2], None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                               line=dict(color=TRACT_COLOR, width=width),
                               name=f"tracts crossing ({len(lines)})",
                               showlegend=True, legendrank=20, hoverinfo="skip"))
    return fig


def run(subject, overlap, *, shell_opacity=0.2, top=None, show=True, show_tracts=True,
        palette="yba"):
    # Tumour and tracts first in the legend, then the shell, then the parcels.
    info = parcels()
    surfaces = meshes()
    index = {p["name"]: i for i, p in enumerate(info)}

    other = "contrast" if palette != "contrast" else "yba"
    here, there = colours(palette), colours(other)

    involved = overlap["parcels"][:top] if top else overlap["parcels"]

    fig = go.Figure()
    add(subject, fig)
    if show_tracts:
        add_tracts(subject, overlap, fig)

    # one closed MNI surface, not the 696 parcel meshes: stacking those piles up
    # alpha until the involved parcels inside are darker than the shell around them
    shell_trace = [len(fig.data)]
    fig.add_trace(_mesh3d(*brain(), color=SHELL_COLOR, opacity=shell_opacity,
                          name="brain surface", showlegend=True,
                          legendrank=30, hoverinfo="skip"))

    # no legendgroup: traces sharing one toggle together, which would make
    # clicking a single parcel hide all of them
    parcel_traces, keep, swap = [], [], []
    for rank, row in enumerate(involved, start=100):
        i = index[row["parcel"]]
        parcel_traces.append(len(fig.data))
        keep.append(f"rgb{here[i]}")
        swap.append(f"rgb{there[i]}")
        fig.add_trace(_mesh3d(
            np.asarray(surfaces[i].vertices), np.asarray(surfaces[i].faces),
            color=f"rgb{here[i]}", opacity=0.95, legendrank=rank,
            name=f"{row['parcel']} ({row['fraction']:.0%})", showlegend=True,
            hovertemplate=f"{row['long_name']}<extra></extra>"))

    fig.update_layout(
        title=f"{subject}, {overlap['tumour_ml']} mL tumour",
        template="plotly_white",
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                   aspectmode="data", domain=dict(x=[0, 0.8], y=[0.1, 0.97])),
        margin=dict(l=0, r=0, b=10, t=50),
        showlegend=True,
        legend=dict(itemsizing="constant", font=dict(size=10), tracegroupgap=6,
                    x=1.0, xanchor="right", y=1.0, yanchor="top",
                    bgcolor="rgba(255,255,255,0.75)"),
        updatemenus=_menus(parcel_traces, keep, swap, palette),
        sliders=_shell_slider(shell_trace, shell_opacity))

    if show:
        fig.show()
    return fig


def _dropdown(x, options, active=0):
    # A dropdown shows its current setting; a button does not.
    return dict(type="dropdown", direction="up", showactive=True, active=active,
                x=x, xanchor="left", y=-0.02, yanchor="top",
                pad=dict(t=0, b=0), font=dict(size=11),
                bgcolor="#ffffff", bordercolor="#c8ccd0",
                buttons=options)


def _menus(parcel_traces, keep, swap, palette):
    # Only what the legend cannot do: bulk parcel visibility, and recolouring.
    if not parcel_traces:
        return []
    return [
        _dropdown(0.42, [
            dict(label=f"{len(parcel_traces)} parcels shown", method="restyle",
                 args=[{"visible": True}, parcel_traces]),
            dict(label="parcels hidden", method="restyle",
                 args=[{"visible": False}, parcel_traces])]),
        _dropdown(0.62, [
            dict(label="atlas colours" if palette != "contrast" else "high contrast",
                 method="restyle", args=[{"color": keep}, parcel_traces]),
            dict(label="high contrast" if palette != "contrast" else "atlas colours",
                 method="restyle", args=[{"color": swap}, parcel_traces])]),
    ]


def _shell_slider(shell, opacity):
    # Slider beats a dropdown here, you want to sweep it and watch.
    if not shell:
        return []
    steps = [round(v / 20, 2) for v in range(21)]
    return [dict(
        active=min(range(len(steps)), key=lambda i: abs(steps[i] - opacity)),
        x=0.02, xanchor="left", y=0, yanchor="top", len=0.34,
        pad=dict(t=0, b=10),
        currentvalue=dict(prefix="shell opacity  ", font=dict(size=11)),
        tickcolor="#c8ccd0", font=dict(size=9),
        steps=[dict(label=f"{int(v * 100)}" if i % 4 == 0 else "",
                    method="restyle", args=[{"opacity": v}, shell])
               for i, v in enumerate(steps)])]


if __name__ == "__main__":
    import json
    import sys

    from paths import RESULTS, subject_id

    if len(sys.argv) != 2:
        sys.exit("usage: ./view.py <subject>   e.g. ./view.py 36")
    results = json.load(open(RESULTS))
    wanted = sys.argv[1]
    if wanted.isdigit():
        wanted = subject_id(int(wanted))
    if wanted not in results:
        sys.exit(f"{wanted} is not in {RESULTS}. have: {', '.join(sorted(results))}")
    run(wanted, results[wanted]["atlas"])
