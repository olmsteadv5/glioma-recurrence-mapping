#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Interactive 3D view: solid tumour inside the Yale Brain Atlas.

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import pyvista as pv
import trimesh

import tracts as tractography
from atlas import colours, label_volume, meshes, parcels
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


def add(subject, fig, opacity=1.0):
    # Add the tumour to a figure that already exists.
    image = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")
    data = np.asarray(image.dataobj).astype(np.int16)
    for value, name in LABELS.items():
        surface = _tumour_surface(data, image.affine, value)
        if surface is None:
            continue
        fig.add_trace(_mesh3d(*surface, color=TISSUE_COLOR[value], opacity=opacity,
                              name=name, showlegend=True,
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
                               showlegend=True, hoverinfo="skip"))
    return fig


def run(subject, overlap, *, shell_opacity=0.2, top=None, show=True, show_tracts=True,
        palette="yba"):
    # Involved parcels coloured, everything else faint, tumour on top.
    info = parcels()
    surfaces = meshes()
    other = "contrast" if palette != "contrast" else "yba"
    palette_colours = colours(palette)
    other_colours = colours(other)
    index = {p["name"]: i for i, p in enumerate(info)}

    involved = overlap["parcels"][:top] if top else overlap["parcels"]
    involved_names = {r["parcel"] for r in involved}

    fig = go.Figure()

    rest = [surfaces[i] for name, i in index.items() if name not in involved_names]
    if rest:
        shell = trimesh.util.concatenate(rest)
        fig.add_trace(_mesh3d(np.asarray(shell.vertices), np.asarray(shell.faces),
                              color=SHELL_COLOR, opacity=shell_opacity,
                              name="atlas (uninvolved)", showlegend=True,
                              hoverinfo="skip"))
    shell_trace = [0] if rest else []

    parcel_traces, swap, keep = [], [], []
    for row in involved:
        mesh = surfaces[index[row["parcel"]]]
        colour = palette_colours[index[row["parcel"]]]
        swap.append(f"rgb{other_colours[index[row['parcel']]]}")
        keep.append(f"rgb{colour}")
        parcel_traces.append(len(fig.data))
        fig.add_trace(_mesh3d(
            np.asarray(mesh.vertices), np.asarray(mesh.faces),
            color=f"rgb{colour}", opacity=0.95,
            name=f"{row['parcel']} ({row['fraction']:.0%})", showlegend=True,
            legendgroup="parcels",
            hovertemplate=f"{row['long_name']}<extra></extra>"))

    first_tumour = len(fig.data)
    add(subject, fig)
    tumour_traces = list(range(first_tumour, len(fig.data)))

    first_tract = len(fig.data)
    if show_tracts:
        add_tracts(subject, overlap, fig)
    tract_traces = list(range(first_tract, len(fig.data)))

    fig.update_layout(
        title=f"{subject}, {overlap['tumour_ml']} mL tumour",
        template="plotly_white",
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                   aspectmode="data", domain=dict(x=[0, 0.82], y=[0.05, 0.95])),
        margin=dict(l=0, r=0, b=60, t=60),
        showlegend=True,
        legend=dict(itemsizing="constant", font=dict(size=10),
                    x=1.0, xanchor="right", y=1.0, yanchor="top",
                    bgcolor="rgba(255,255,255,0.75)"),
        updatemenus=_controls(shell_trace, parcel_traces, tumour_traces,
                              tract_traces,
                              _palette_button(parcel_traces, swap, keep)),
        sliders=_shell_slider(shell_trace, shell_opacity))

    if show:
        fig.show()
    return fig


def _toggle(label, traces):
    # args2 makes this a real toggle: first click hides, second shows.
    return dict(label=label, method="restyle",
                args=[{"visible": False}, traces],
                args2=[{"visible": True}, traces])


def _palette_button(parcel_traces, other, current):
    # restyle takes a list, one colour per trace, so all parcels swap at once.
    if not parcel_traces:
        return []
    return [dict(label="recolour", method="restyle",
                 args=[{"color": other}, parcel_traces],
                 args2=[{"color": current}, parcel_traces])]


def _controls(shell, parcels_, tumour, tracts, palette_button):
    # One row of toggles, fixed width per button so nothing reflows on click.
    groups = [("parcels", parcels_), ("tumour", tumour), ("tracts", tracts),
              ("shell", shell)]
    buttons = [_toggle(name, traces) for name, traces in groups if traces]
    buttons += palette_button
    if not buttons:
        return []
    return [dict(type="buttons", direction="right", showactive=True,
                 buttons=buttons,
                 x=0.01, xanchor="left", y=1.0, yanchor="top",
                 pad=dict(l=0, r=0, t=0, b=0),
                 bgcolor="#ffffff", bordercolor="#c8ccd0", borderwidth=1,
                 font=dict(size=11))]


def _shell_slider(shell, opacity):
    # Slider beats a dropdown here, you want to sweep it and watch.
    if not shell:
        return []
    steps = [round(v / 20, 2) for v in range(21)]
    return [dict(
        active=min(range(len(steps)), key=lambda i: abs(steps[i] - opacity)),
        x=0.01, xanchor="left", y=0, yanchor="top", len=0.35,
        pad=dict(t=10, b=10),
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
