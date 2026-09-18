# The tumor's neighborhood as a node chart: parcels, and the tracts between them.

import csv
import functools

import nibabel as nib
import numpy as np

from atlas import ATLAS_DIR, ensure, label_volume, parcels
from connect import FILE
from paths import MNI, out_dir

# SC_Length is clamped: over half the edges sit at its maximum. SC_EuclDist is
# the real center to center distance, so that is what sets the drawn lengths.
# a parcel joins the chart once this many tracts through the tumor reach it
LEAST = 3


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


@functools.lru_cache(maxsize=4)
def through(subject):
    # Every centerline that runs through the tumor: which parcels it reaches,
    # how far along itself from the mass to each of them, and which pairs of
    # parcels it therefore joins. This is the whole of what the chart draws.
    import collections
    import itertools
    import tracts as tractography

    labels, names = label_volume(nib.load(MNI))
    seg = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")
    tumor = np.asarray(seg.dataobj) > 0
    inverse = np.linalg.inv(seg.affine)
    shape = np.array(tumor.shape)

    reach = collections.defaultdict(list)     # parcel -> [(tract, mm to the mass)]
    joins = collections.defaultdict(list)     # parcel pair -> [(tract, mm apart)]
    skirt = collections.defaultdict(list)     # parcel -> [(tract, mm)] missing the mass
    crossing, stray = 0, []
    for n, line in enumerate(tractography.centerlines()):
        ijk = np.rint(nib.affines.apply_affine(inverse, line)).astype(int)
        inside = np.all((ijk >= 0) & (ijk < shape), axis=1)
        keep = ijk[inside]
        if not len(keep):
            continue
        at = labels[keep[:, 0], keep[:, 1], keep[:, 2]]
        points = line[inside]
        walked = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(points, axis=0), axis=1))])
        middles = {}
        for value in np.unique(at):
            if value <= 0:
                continue
            middles[int(value) - 1] = int(np.median(np.where(at == value)[0]))

        if not tumor[keep[:, 0], keep[:, 1], keep[:, 2]].any():
            # it misses the mass, so it is nothing to the connectome, but it
            # does run through the tissue the tumor has taken
            for parcel in middles:
                skirt[parcel].append((n, float(walked[-1])))
            continue
        crossing += 1
        mass = int(np.median(np.where(tumor[keep[:, 0], keep[:, 1], keep[:, 2]])[0]))
        if not middles:
            stray.append((n, float(walked[-1])))
            continue
        for parcel, where in middles.items():
            reach[parcel].append((n, max(abs(float(walked[mass] - walked[where])), 1.0)))
        for a, b in itertools.combinations(sorted(middles), 2):
            joins[(a, b)].append(
                (n, max(abs(float(walked[middles[a]] - walked[middles[b]])), 1.0)))
    return dict(reach), dict(joins), dict(skirt), crossing, stray


def build(subject, overlap, least=LEAST):
    # The connectome the tumor sits in: parcels as nodes, the tumor as one more,
    # and an edge wherever a tract that runs through the tumor ties two of them.
    names = [p["name"] for p in parcels()]
    index = {name: i for i, name in enumerate(names)}
    inside = {index[r["parcel"]] for r in overlap["parcels"] if r["parcel"] in index}
    reach, joins, skirt, crossing, stray = through(subject)

    # a parcel one stray tract grazes is noise on a plan; a parcel a dozen run
    # into is the point of the picture, so a parcel earns its place by count
    nodes = sorted({p for p, hits in reach.items() if len(hits) >= least} | inside)
    place = {n: k for k, n in enumerate(nodes)}
    middle, tumor = centers(subject)

    def gap(a, b):
        return max(float(np.linalg.norm(middle[a] - middle[b])), 1.0)

    link = []
    for (a, b), hits in joins.items():
        if a in place and b in place:
            link.append((place[a], place[b], len(hits), gap(a, b),
                         [mm for _, mm in hits], [n for n, _ in hits]))

    to_tumor = np.array([float(np.linalg.norm(middle[n] - tumor))
                         if not np.isnan(middle[n]).any() else 60.0 for n in nodes])
    spokes = [(place[n], len(reach[n]), [mm for _, mm in reach[n]],
               [t for t, _ in reach[n]]) for n in nodes if n in reach]

    # a second hub for everything whose far end the atlas cannot name: the
    # tracts that leave the mass and land nowhere, and the ones that miss the
    # mass but run through the tissue it has taken
    clipped = sorted(inside & set(skirt))
    void = [(None, len(stray), [mm for _, mm in stray], [t for t, _ in stray])]
    void += [(place[n], len(skirt[n]), [mm for _, mm in skirt[n]],
              [t for t, _ in skirt[n]]) for n in clipped if n in place]
    # a tract clipping two involved parcels is one tract on two lines, so the
    # total the chart accounts for counts it once
    clips = len({t for n in clipped for t, _ in skirt[n]})
    return nodes, link, to_tumor, inside, spokes, void, crossing + clips


def attach(subject, nodes, parcel, least=1):
    # One more parcel for a chart already built, for a connectivity row naming
    # something too lightly touched to have earned a node of its own.
    names = [p["name"] for p in parcels()]
    if parcel not in names or names.index(parcel) in nodes:
        return None
    n = names.index(parcel)
    reach, joins, _, _, _ = through(subject)
    middle, tumor = centers(subject)
    seat = {v: k for k, v in enumerate(nodes)}

    link = []
    for (a, b), hits in joins.items():
        other = b if a == n else a if b == n else None
        if other is None or other not in seat or len(hits) < least:
            continue
        far = max(float(np.linalg.norm(middle[n] - middle[other])), 1.0)
        link.append((seat[other], len(hits), far,
                     [mm for _, mm in hits], [t for t, _ in hits]))
    mine = reach.get(n, [])
    far = (float(np.linalg.norm(middle[n] - tumor))
           if not np.isnan(middle[n]).any() else 60.0)
    return n, link, far, [mm for _, mm in mine], [t for t, _ in mine]
