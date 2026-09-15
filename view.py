# Interactive 3D view: solid tumour inside the Yale Brain Atlas.

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import pyvista as pv
import trimesh

import tracts as tractography
from atlas import label_volume, meshes, parcels
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


def run(subject, overlap, *, shell_opacity=0.2, top=30, show=True, show_tracts=True):
    # Involved parcels coloured, everything else faint, tumour on top.
    info = parcels()
    surfaces = meshes()
    index = {p["name"]: i for i, p in enumerate(info)}

    involved = overlap["parcels"][:top]
    involved_names = {r["parcel"] for r in involved}

    fig = go.Figure()

    rest = [surfaces[i] for name, i in index.items() if name not in involved_names]
    if rest:
        shell = trimesh.util.concatenate(rest)
        fig.add_trace(_mesh3d(np.asarray(shell.vertices), np.asarray(shell.faces),
                              color=SHELL_COLOR, opacity=shell_opacity,
                              name="atlas (uninvolved)", showlegend=True,
                              hoverinfo="skip"))

    for row in involved:
        mesh = surfaces[index[row["parcel"]]]
        colour = info[index[row["parcel"]]]["color"]
        fig.add_trace(_mesh3d(
            np.asarray(mesh.vertices), np.asarray(mesh.faces),
            color=f"rgb{colour}", opacity=0.95,
            name=f"{row['parcel']} ({row['fraction']:.0%})", showlegend=True,
            hovertemplate=f"{row['long_name']}<extra></extra>"))

    add(subject, fig)
    if show_tracts:
        add_tracts(subject, overlap, fig)

    fig.update_layout(
        title=f"{subject}, {overlap['tumour_ml']} mL tumour",
        template="plotly_white",
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                   aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(itemsizing="constant", font=dict(size=10)))

    if show:
        fig.show()
    return fig
