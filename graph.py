# The tumour's neighbourhood as a node chart: parcels, and the tracts between them.

import csv
import functools

import nibabel as nib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from atlas import ATLAS_DIR, ensure, label_volume, parcels
from connect import FILE
from paths import MNI, out_dir

# SC_Length is clamped: over half the edges sit at its maximum. SC_EuclDist is
# the real centre to centre distance, so that is what sets the drawn lengths.
FLOOR = 0.15


@functools.lru_cache(maxsize=1)
def edges():
    # Connection strength and distance for every parcel pair, as two matrices.
    cache = ATLAS_DIR / "graph_edges.npz"
    if cache.exists():
        held = np.load(cache)
        return held["count"], held["distance"]

    names = [p["name"] for p in parcels()]
    index = {name: i for i, name in enumerate(names)}
    count = np.zeros((len(names), len(names)), dtype=np.float32)
    distance = np.zeros_like(count)
    with open(ensure(FILE)) as fh:
        for row in csv.DictReader(fh):
            a, b = index.get(row["Parcel_A_name"]), index.get(row["Parcel_B_name"])
            if a is None or b is None or not row.get("SC_Count"):
                continue
            count[a, b] = count[b, a] = float(row["SC_Count"])
            if row.get("SC_EuclDist"):
                distance[a, b] = distance[b, a] = float(row["SC_EuclDist"])

    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, count=count, distance=distance)
    return count, distance


@functools.lru_cache(maxsize=4)
def centres(subject):
    # Every parcel's middle, and the tumour's, in MNI millimetres.
    labels, names = label_volume(nib.load(MNI))
    affine = nib.load(MNI).affine
    middle = np.full((len(names), 3), np.nan, dtype=np.float32)
    found = np.nonzero(labels)
    order = np.argsort(labels[found])
    voxels = np.stack(found, axis=1)[order]
    values = labels[found][order]
    starts = np.searchsorted(values, np.arange(1, len(names) + 1), side="left")
    stops = np.searchsorted(values, np.arange(1, len(names) + 1), side="right")
    for i, (a, b) in enumerate(zip(starts, stops)):
        if b > a:
            middle[i] = nib.affines.apply_affine(affine, voxels[a:b].mean(0))

    seg = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")
    hit = np.argwhere(np.asarray(seg.dataobj) > 0)
    tumour = nib.affines.apply_affine(seg.affine, hit.mean(0)) if len(hit) else np.zeros(3)
    return middle, tumour


def build(subject, overlap, floor=FLOOR):
    # Involved parcels, plus whatever the white matter ties them to.
    count, distance = edges()
    names = [p["name"] for p in parcels()]
    index = {name: i for i, name in enumerate(names)}
    inside = [index[r["parcel"]] for r in overlap["parcels"] if r["parcel"] in index]
    if not inside:
        return [], [], np.zeros((0, 2))

    near = set(inside)
    for i in inside:
        near |= set(np.where(count[i] >= floor)[0].tolist())
    nodes = sorted(near)
    place = {n: k for k, n in enumerate(nodes)}

    link = []
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            weight = count[nodes[a], nodes[b]]
            if weight >= floor:
                link.append((a, b, float(weight), float(distance[nodes[a], nodes[b]])))

    middle, tumour = centres(subject)
    # the tumour is the last node, its distance to each parcel being the real one
    to_tumour = [float(np.linalg.norm(middle[n] - tumour))
                 if not np.isnan(middle[n]).any() else 60.0 for n in nodes]
    return nodes, link, np.array(to_tumour), set(inside)


def layout(count_of_nodes, link, to_tumour, rounds=120):
    # Stress majorisation: put the nodes where the drawn gaps match the real ones.
    total = count_of_nodes + 1
    target = np.zeros((total, total), dtype=np.float32)
    for a, b, _, far in link:
        target[a, b] = target[b, a] = far or 60.0
    target[:count_of_nodes, -1] = target[-1, :count_of_nodes] = to_tumour

    known = csr_matrix(np.where(target > 0, target, 0))
    full = shortest_path(known, directed=False)
    limit = np.nanmax(full[np.isfinite(full)]) if np.isfinite(full).any() else 100.0
    full[~np.isfinite(full)] = limit * 1.4
    np.fill_diagonal(full, 0.0)

    rng = np.random.default_rng(0)
    points = rng.normal(0, limit / 4, (total, 2))
    weight = np.where(full > 0, 1.0 / np.maximum(full, 1e-6) ** 2, 0.0)
    scale = weight.sum(1)
    for _ in range(rounds):
        gap = np.linalg.norm(points[:, None] - points[None], axis=2)
        ratio = np.divide(full, np.maximum(gap, 1e-6), out=np.zeros_like(full), where=gap > 0)
        step = (weight * ratio)[:, :, None] * (points[:, None] - points[None])
        points = (weight @ points + step.sum(1)) / np.maximum(scale, 1e-9)[:, None]
    return points
