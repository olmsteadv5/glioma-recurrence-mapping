# White matter tract centrelines from the official top-5000 tract mesh.

import numpy as np

from atlas import ensure
from paths import ATLAS_DIR

CACHE = ATLAS_DIR / "tract_centrelines.npz"

# the OBJ is Y-up Blender; atlas space is (x, -z, y)
AXES = (0, 2, 1)
SIGNS = np.array([1.0, -1.0, 1.0])

MIN_VERTICES = 12
STEP_MM = 4.0


def _parse_obj(path):
    vertices, faces = [], []
    for line in open(path):
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.array(vertices)[:, AXES] * SIGNS, np.array(faces)


def _centreline(points, step=STEP_MM):
    # Mean position in bins along the bundle's principal axis.
    centre = points.mean(0)
    axis = np.linalg.svd(points - centre, full_matrices=False)[2][0]
    t = (points - centre) @ axis
    span = t.max() - t.min()
    if span < step:
        return None
    bins = max(int(span / step), 2)
    index = np.clip(((t - t.min()) / span * bins).astype(int), 0, bins - 1)
    line = [points[index == b].mean(0) for b in range(bins) if (index == b).any()]
    return np.array(line) if len(line) >= 2 else None


def centrelines():
    # One polyline per connected bundle, cached.
    if CACHE.exists():
        data = np.load(CACHE)
        return np.split(data["points"], data["offsets"][1:-1])

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    vertices, faces = _parse_obj(ensure("top5k_tracts_decimated.obj"))
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(len(vertices), len(vertices)))
    _, label = connected_components(graph, directed=False)

    lines = []
    for component in np.where(np.bincount(label) >= MIN_VERTICES)[0]:
        line = _centreline(vertices[label == component])
        if line is not None:
            lines.append(line)

    offsets = np.cumsum([0] + [len(l) for l in lines])
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, points=np.vstack(lines), offsets=offsets)
    return lines


def crossing(lines, seg_img, labels, involved):
    # Tracts passing through the tumour or any involved parcel.
    import nibabel as nib

    tumour = np.asarray(seg_img.dataobj) > 0
    shape = np.array(tumour.shape)
    inverse = np.linalg.inv(seg_img.affine)
    wanted = set(involved)

    hits = []
    for line in lines:
        ijk = np.rint(nib.affines.apply_affine(inverse, line)).astype(int)
        inside = np.all((ijk >= 0) & (ijk < shape), axis=1)
        if not inside.any():
            continue
        sel = ijk[inside]
        if tumour[sel[:, 0], sel[:, 1], sel[:, 2]].any():
            hits.append(line)
            continue
        if wanted and np.isin(labels[sel[:, 0], sel[:, 1], sel[:, 2]], list(wanted)).any():
            hits.append(line)
    return hits
