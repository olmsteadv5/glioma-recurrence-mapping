"""Tumour-in-atlas 3D view, following the project's add/run contract.

    add(subject, viz, ...)       -> appends tumour traces to the current figure
    run(subject, yba, viz, ...)  -> new() + parcels + tumour + show()

`yba` is the YaleBrainAtlas, `viz` its YBAVisualizer.
"""

import nibabel as nib
import numpy as np
import pyvista as pv

from paths import LABELS, out_dir

TISSUE_COLOR = {1: "#f2c14e", 2: "#3fa7d6", 3: "#e4572e", 4: "#8e7dbe"}


def _surface(data, affine, value):
    """Marching-cubes surface for one tissue class, in MNI millimetres."""
    binary = (data == value).astype(np.float32)
    if binary.sum() < 10:
        return None
    grid = pv.ImageData(dimensions=np.array(binary.shape) + 1)
    grid.cell_data["v"] = binary.flatten(order="F")
    surface = grid.cells_to_points("v").contour([0.5], scalars="v")
    if surface.n_points == 0:
        return None
    surface = surface.smooth(n_iter=30, relaxation_factor=0.1)
    vertices = nib.affines.apply_affine(affine, surface.points)
    return vertices, np.asarray(surface.faces).reshape(-1, 4)[:, 1:]


def add(subject, viz, opacity=1.0):
    """Append one solid mesh per tumour tissue class to the current figure."""
    image = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")
    data = np.asarray(image.dataobj).astype(np.int16)

    for value, name in LABELS.items():
        mesh = _surface(data, image.affine, value)
        if mesh is None:
            continue
        vertices, faces = mesh
        viz.add_custom_mesh(vertices, faces, color=TISSUE_COLOR[value],
                            opacity=opacity, label=name)


def run(subject, yba, viz, result, *, figure_name=None, shell_opacity=0.2, top=30):
    """Fresh figure: involved parcels coloured by involvement, tumour solid on top."""
    figure_name = figure_name or subject
    involved = [row["parcel"] for row in result["parcels"]][:top]
    fractions = {row["parcel"]: row["fraction"] for row in result["parcels"]}

    values = [fractions.get(name, 0.0) for name in yba.parcel_names]
    viz.new(figure_name)
    # labels must be passed explicitly; add_parcels builds hover text wrongly without it
    viz.add_parcels(values, labels=values, segment=set(involved),
                    opacity=1.0 - shell_opacity, colorscale="reds",
                    title="Fraction in tumour")
    add(subject, viz)

    tumour, in_parcels = result["tumour_ml"], result["in_parcels_ml"]
    viz.update_layout(figure_name, title=(
        f"{subject} - {tumour:.0f} mL tumour, {in_parcels:.0f} mL "
        f"({in_parcels / tumour:.0%}) in parcelled tissue, {len(involved)} parcels"))
    return viz.show(figure_name)
