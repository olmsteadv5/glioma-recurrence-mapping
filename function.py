# Parcelsynth: which functions each parcel is associated with, and what a tumour threatens.

import csv
import functools

import numpy as np

from atlas import ensure, parcels

FILES = {"functional": "parcelsynth_activations_functionalterms.csv",
         "all": "parcelsynth_activations_allterms.csv"}


@functools.lru_cache(maxsize=2)
def terms(kind="functional"):
    # Term names and a 696 x N score matrix, rows in parcel order.
    with open(ensure(FILES[kind])) as fh:
        rows = list(csv.reader(fh))
    names = [p["name"] for p in parcels()]
    scores = np.array([[float(x) for x in r] for r in rows[1:]], dtype=np.float32)
    if len(scores) != len(names):
        raise RuntimeError(f"{FILES[kind]} has {len(scores)} rows, atlas has {len(names)}")
    return rows[0], scores, names


def for_parcel(parcel, n=8, kind="functional"):
    # A single parcel's strongest terms.
    words, scores, names = terms(kind)
    row = scores[names.index(parcel)]
    order = np.argsort(-row)[:n]
    return [(words[i], round(float(row[i]), 2)) for i in order if row[i] > 0]


def for_tumour(overlap, n=20, kind="functional"):
    # Each involved parcel's profile, weighted by how much of that parcel is destroyed.
    words, scores, names = terms(kind)
    index = {name: i for i, name in enumerate(names)}

    weights = np.zeros(len(names), dtype=np.float32)
    for row in overlap["parcels"]:
        if row["parcel"] in index:
            weights[index[row["parcel"]]] = row["fraction"]
    if weights.sum() == 0:
        return []

    total = (scores * weights[:, None]).sum(0) / weights.sum()
    order = np.argsort(-total)[:n]
    return [{"term": words[i], "score": round(float(total[i]), 2)}
            for i in order if total[i] > 0]
