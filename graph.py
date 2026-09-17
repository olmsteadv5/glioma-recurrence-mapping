# The tumor's neighborhood as a node chart: parcels, and the tracts between them.

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
# the real center to center distance, so that is what sets the drawn lengths.
FLOOR = 0.15


def sample(lengths):
    # Every tract, longest last. There used to be a cap of eight a parcel here,
    # which quietly hid 182 of L_CC_B's tracts behind a drawn 8.
    return sorted(float(v) for v in lengths)


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
def centers(subject):
    # Every parcel's middle, and the tumor's, in MNI millimeters.
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
    tumor = nib.affines.apply_affine(seg.affine, hit.mean(0)) if len(hit) else np.zeros(3)
    return middle, tumor


@functools.lru_cache(maxsize=1)
def bundles():
    # Every centerline's real length, filed under the parcels it runs between.
    # Tracts curve: the median one is 2.2 times the straight line it spans, and
    # two tracts joining the same pair differ in length by about a fifth.
    import collections
    import itertools
    import tracts as tractography

    cache = ATLAS_DIR / "graph_bundles_middles.npz"
    if cache.exists():
        held = np.load(cache)
        return {tuple(k): v for k, v in zip(held["pairs"], np.split(
            held["lengths"], held["offsets"][1:-1]))}

    labels, names = label_volume(nib.load(MNI))
    mni = nib.load(MNI)
    inverse = np.linalg.inv(mni.affine)
    shape = np.array(labels.shape)

    runs = collections.defaultdict(list)
    for line in tractography.centerlines():
        ijk = np.rint(nib.affines.apply_affine(inverse, line)).astype(int)
        keep = np.all((ijk >= 0) & (ijk < shape), axis=1)
        if keep.sum() < 2:
            continue
        points = line[keep]
        sel = ijk[keep]
        at = labels[sel[:, 0], sel[:, 1], sel[:, 2]]
        walked = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(points, axis=0), axis=1))])
        seen = [int(v) - 1 for v in np.unique(at) if v > 0]
        for a, b in itertools.combinations(sorted(seen), 2):
            # middle of the run in one parcel to middle of the run in the other,
            # to match the center to center gap the chart draws it across. the
            # nearest touching points instead would measure edge to edge, which
            # is near zero for parcels that share a border
            here, there = np.where(at == a + 1)[0], np.where(at == b + 1)[0]
            runs[(a, b)].append(abs(float(walked[int(np.median(here))]
                                          - walked[int(np.median(there))])))

    pairs = sorted(runs)
    lengths = np.concatenate([runs[k] for k in pairs]) if pairs else np.zeros(0)
    offsets = np.cumsum([0] + [len(runs[k]) for k in pairs])
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, pairs=np.array(pairs), lengths=lengths, offsets=offsets)
    return {k: np.array(runs[k]) for k in pairs}


@functools.lru_cache(maxsize=4)
def through(subject):
    # How many tract centerlines run through both the tumor and each parcel.
    import collections
    import tracts as tractography

    labels, names = label_volume(nib.load(MNI))
    seg = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")
    tumor = np.asarray(seg.dataobj) > 0
    inverse = np.linalg.inv(seg.affine)
    shape = np.array(tumor.shape)

    import itertools

    hits, crossing, landed, stray = collections.defaultdict(list), set(), [], []
    for line in tractography.centerlines():
        ijk = np.rint(nib.affines.apply_affine(inverse, line)).astype(int)
        keep = ijk[np.all((ijk >= 0) & (ijk < shape), axis=1)]
        if not len(keep) or not tumor[keep[:, 0], keep[:, 1], keep[:, 2]].any():
            continue
        at = labels[keep[:, 0], keep[:, 1], keep[:, 2]]
        points = line[np.all((ijk >= 0) & (ijk < shape), axis=1)]
        walked = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(points, axis=0), axis=1))])
        mass = np.where(tumor[keep[:, 0], keep[:, 1], keep[:, 2]])[0]
        middle = int(np.median(mass))
        for value in np.unique(at):
            if value <= 0:
                continue
            there = np.where(at == value)[0]
            hits[int(value) - 1].append(
                max(abs(float(walked[middle] - walked[int(np.median(there))])), 1.0))
        # this tract crosses the mass, so every pair it joins is affected
        reached = sorted(int(v) - 1 for v in np.unique(at) if v > 0)
        crossing.update(itertools.combinations(reached, 2))
        # kept per tract as well, so a chart can say how many of them it stands for
        landed.append(frozenset(reached))
        if not reached:
            # through the mass and into nothing the atlas names. it still ran
            # somewhere, so the chart owes it a mark rather than a silent drop
            stray.append(float(walked[-1]))
    return ({k: (len(v), v) for k, v in hits.items()}, crossing, landed, stray)


def build(subject, overlap, floor=FLOOR):
    # Involved parcels, plus whatever the white matter ties them to.
    count, distance = edges()
    names = [p["name"] for p in parcels()]
    index = {name: i for i, name in enumerate(names)}
    inside = [index[r["parcel"]] for r in overlap["parcels"] if r["parcel"] in index]
    if not inside:
        return [], [], np.zeros((0, 2))

    reach, crossing, _, stray = through(subject)
    near = set(inside) | set(reach)
    for i in inside:
        near |= set(np.where(count[i] >= floor)[0].tolist())
    nodes = sorted(near)
    place = {n: k for k, n in enumerate(nodes)}

    runs, link = bundles(), []
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            key = (min(nodes[a], nodes[b]), max(nodes[a], nodes[b]))
            weight, real = count[nodes[a], nodes[b]], runs.get(key)
            # a measured tract earns a line whatever the atlas says about the
            # pair, and the atlas earns one on its own above the floor
            if real is None and weight < floor:
                continue
            straight = float(distance[nodes[a], nodes[b]]) or 1.0
            ratios = [1.0] if real is None else sample(v / straight for v in real)
            kind = ("via" if key in crossing else "clear" if real is not None
                    else "atlas")
            link.append((a, b, float(weight), straight, ratios, kind))

    middle, tumor = centers(subject)
    # the tumor is the last node, its distance to each parcel being the real one
    to_tumor = [float(np.linalg.norm(middle[n] - tumor))
                 if not np.isnan(middle[n]).any() else 60.0 for n in nodes]
    # and its edges are real: centerlines that pass through the mass and the parcel
    spokes = []
    for n in nodes:
        if not reach.get(n):
            continue
        near = max(float(np.linalg.norm(middle[n] - tumor)), 1.0) \
            if not np.isnan(middle[n]).any() else 60.0
        spokes.append((place[n], reach[n][0],
                       sample(v / near for v in reach[n][1]) or [1.0]))
    return nodes, link, np.array(to_tumor), set(inside), spokes, stray


def layout(count_of_nodes, link, to_tumor, rounds=120):
    # Stress majorization: put the nodes where the drawn gaps match the real ones.
    total = count_of_nodes + 1
    target = np.zeros((total, total), dtype=np.float32)
    for a, b, _, far, _, _ in link:
        target[a, b] = target[b, a] = far or 60.0
    target[:count_of_nodes, -1] = target[-1, :count_of_nodes] = to_tumor

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


def attach(subject, nodes, parcel, most=12):
    # One more parcel for a chart already built: a connectivity row can name
    # something too weakly tied to have made the node set, and a row that
    # highlights nothing is worse than a weak edge drawn honestly.
    count, distance = edges()
    names = [p["name"] for p in parcels()]
    if parcel not in names or names.index(parcel) in nodes:
        return None
    n = names.index(parcel)
    runs, (_, crossing, _, _) = bundles(), through(subject)
    order = sorted(range(len(nodes)), key=lambda k: -count[n, nodes[k]])[:most]

    link = []
    for k in order:
        weight = float(count[n, nodes[k]])
        if weight <= 0:
            continue
        straight = float(distance[n, nodes[k]]) or 1.0
        key = (min(n, nodes[k]), max(n, nodes[k]))
        real = runs.get(key)
        ratios = [1.0] if real is None else sample(v / straight for v in real)
        kind = "via" if key in crossing else "clear" if real is not None else "atlas"
        link.append((k, weight, straight, ratios, kind))

    middle, tumor = centers(subject)
    far = (float(np.linalg.norm(middle[n] - tumor))
           if not np.isnan(middle[n]).any() else 60.0)
    return n, link, far


def settle(points, link, far, rounds=60):
    # Where that one node goes among nodes already placed: the same stress rule
    # as the layout, with everything but this one held still.
    fixed = np.array([points[k] for k, *_ in link] + [points[-1]])
    target = np.array([edge[2] for edge in link] + [far], dtype=float)
    weight = 1.0 / np.maximum(target, 1e-6) ** 2
    at = fixed.mean(0) + 1.0
    for _ in range(rounds):
        gap = np.maximum(np.linalg.norm(at - fixed, axis=1), 1e-6)
        pull = fixed + (at - fixed) * (target / gap)[:, None]
        at = (weight[:, None] * pull).sum(0) / weight.sum()
    return at
