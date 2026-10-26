
import functools

import nibabel as nib
import numpy as np

from yale.atlas import label_volume, parcels
from paths import MNI, out_dir

# decision: a parcel joins the chart at this share of tumor streamlines, never under three. judgement call
LEAST = 0.005
FEWEST = 3


@functools.lru_cache(maxsize=4)
def centers(subject):
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
    # explainer: every streamline through the tumor, which parcels it reaches and how far along it, and which pairs it joins
    import collections
    import itertools
    from hcp import streamlines

    labels, names = label_volume(nib.load(MNI))
    seg = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")
    tumor = np.asarray(seg.dataobj) > 0
    inverse = np.linalg.inv(seg.affine)
    shape = np.array(tumor.shape)

    reach = collections.defaultdict(list)  # explainer: parcel -> [(tract, mm to the mass)]
    joins = collections.defaultdict(list)  # explainer: parcel pair -> [(tract, mm apart)]
    skirt = collections.defaultdict(list)  # explainer: parcel -> [(tract, mm)] missing the mass
    crossing, stray = 0, []
    for n, line in enumerate(streamlines.streamlines()):
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
            # explainer: misses the mass but runs through tissue the tumor took
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
    # explainer: parcels as nodes plus the tumor, an edge wherever a tumor tract ties two
    names = [p["name"] for p in parcels()]
    index = {name: i for i, name in enumerate(names)}
    inside = {index[r["parcel"]] for r in overlap["parcels"] if r["parcel"] in index}
    reach, joins, skirt, crossing, stray = through(subject)

    # decision: a parcel earns its place by tract count, one stray tract is noise
    floor = max(least * crossing, FEWEST)
    nodes = sorted({p for p, hits in reach.items() if len(hits) >= floor} | inside)
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

    # decision: a second hub for far ends the atlas can't name, and for tracts that miss the mass but cross taken tissue
    clipped = sorted(inside & set(skirt))
    void = [(None, len(stray), [mm for _, mm in stray], [t for t, _ in stray])]
    void += [(place[n], len(skirt[n]), [mm for _, mm in skirt[n]],
              [t for t, _ in skirt[n]]) for n in clipped if n in place]
    # decision: a tract clipping two parcels is one tract, counted once in the total
    clips = len({t for n in clipped for t, _ in skirt[n]})
    return nodes, link, to_tumor, inside, spokes, void, crossing + clips


def attach(subject, nodes, parcel, least=1):
    # explainer: adds a parcel to a built chart, for a connectivity row too lightly touched to have a node
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
