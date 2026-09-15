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


def for_tumour(overlap, n=15, kind="functional", floor=0.15):
    # Terms over-represented in the damaged tissue, not just terms that score high everywhere.
    words, scores, names = terms(kind)
    index = {name: i for i, name in enumerate(names)}

    weights = np.zeros(len(names), dtype=np.float32)
    for row in overlap["parcels"]:
        if row["parcel"] in index:
            weights[index[row["parcel"]]] = row["fraction"]
    if weights.sum() == 0:
        return []

    lesion = (scores * weights[:, None]).sum(0) / weights.sum()
    baseline = scores.mean(0)
    # ratio against the whole brain, so common terms do not win every time
    enrichment = lesion / np.maximum(baseline, 1e-6)
    # a term barely present in the lesion can still divide to a huge ratio; drop those
    enrichment[lesion < floor * lesion.max()] = 0

    order = np.argsort(-enrichment)[:n]
    return [{"term": words[i], "enrichment": round(float(enrichment[i]), 2),
             "score": round(float(lesion[i]), 3)} for i in order if enrichment[i] > 0]
