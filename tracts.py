# White matter tract centerlines from the official top-5000 tract mesh.

import numpy as np

from atlas import ensure
from paths import ATLAS_DIR

CACHE = ATLAS_DIR / "tract_centerlines_geodesic.npz"

# the OBJ is Y-up, atlas space is (x, -z, y)
AXES = (0, 2, 1)
SIGNS = np.array([1.0, -1.0, 1.0])

MIN_VERTICES = 12
# one step per voxel, so a line cannot straddle a parcel without landing in it
STEP_MM = 1.0


def _parse_obj(path):
    vertices, faces = [], []
    for line in open(path):
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.array(vertices)[:, AXES] * SIGNS, np.array(faces)


def _smooth(line, rounds=2):
    # the mesh is coarse enough that a 1 mm slice often holds a single surface
    # vertex, so the raw line zigzags round the tube and reads 9% long
    for _ in range(rounds):
        line = line.copy()
        line[1:-1] = (line[:-2] + 2 * line[1:-1] + line[2:]) / 4
    return line


def _resample(line, step=STEP_MM):
    # even steps along the path, so measuring and voxel lookups both walk it
    walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    if walked[-1] < step:
        return line
    want = np.arange(0, walked[-1], step)
    return np.stack([np.interp(want, walked, line[:, k]) for k in range(3)], axis=1)


def _centerline(points, edges, step=STEP_MM):
    # Order the tube by distance along its own surface, not by a straight axis.
    # A hooked bundle has both limbs at the same place on its long axis, so
    # projecting onto that axis averages them together and cuts the corner.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    mesh = coo_matrix((length, (edges[:, 0], edges[:, 1])), shape=(len(points),) * 2)
    far = dijkstra(mesh, directed=False, indices=0)
    end = int(np.nanargmax(np.where(np.isfinite(far), far, -1)))
    along = dijkstra(mesh, directed=False, indices=end)
    along[~np.isfinite(along)] = 0.0

    span = along.max()
    if span < step:
        return None
    bins = max(int(span / step), 2)
    index = np.clip((along / span * bins).astype(int), 0, bins - 1)
    line = [points[index == b].mean(0) for b in range(bins) if (index == b).any()]
    if len(line) < 3:
        return np.array(line) if len(line) >= 2 else None
    return _resample(_smooth(np.array(line)))


def centerlines():
    # One line per bundle. Slow to build, so cache it.
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
        where = np.where(label == component)[0]
        place = np.full(len(vertices), -1)
        place[where] = np.arange(len(where))
        mine = (place[edges[:, 0]] >= 0) & (place[edges[:, 1]] >= 0)
        line = _centerline(vertices[where], place[edges[mine]])
        if line is not None:
            lines.append(line)

    offsets = np.cumsum([0] + [len(l) for l in lines])
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, points=np.vstack(lines), offsets=offsets)
    return lines


def crossing(lines, seg_img, labels, involved):
    # Lines through the mass, and lines that miss it but clip an involved parcel.
    import nibabel as nib

    tumor = np.asarray(seg_img.dataobj) > 0
    shape = np.array(tumor.shape)
    inverse = np.linalg.inv(seg_img.affine)
    wanted = list(set(involved))

    mass, nearby = [], []
    for line in lines:
        ijk = np.rint(nib.affines.apply_affine(inverse, line)).astype(int)
        inside = np.all((ijk >= 0) & (ijk < shape), axis=1)
        if not inside.any():
            continue
        sel = ijk[inside]
        if tumor[sel[:, 0], sel[:, 1], sel[:, 2]].any():
            mass.append(line)
        elif wanted and np.isin(labels[sel[:, 0], sel[:, 1], sel[:, 2]], wanted).any():
            nearby.append(line)
    return mass, nearby
