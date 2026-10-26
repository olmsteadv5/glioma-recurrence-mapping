# explainer: Parcelsynth functions per parcel, and what a tumor puts at risk

import csv
import functools

import numpy as np

from yale.atlas import ensure, parcels

FILES = {"functional": "parcelsynth_activations_functionalterms.csv",
         "all": "parcelsynth_activations_allterms.csv"}


@functools.lru_cache(maxsize=2)
def terms(kind="functional"):
    # explainer: term names and a parcel x term score matrix, rows in parcel order
    with open(ensure(FILES[kind])) as fh:
        rows = list(csv.reader(fh))
    names = [p["name"] for p in parcels()]
    scores = np.array([[float(x) for x in r] for r in rows[1:]], dtype=np.float32)
    if len(scores) != len(names):
        raise RuntimeError(f"{FILES[kind]} has {len(scores)} rows, atlas has {len(names)}")
    return rows[0], scores, names


def for_parcel(parcel, n=8, kind="functional"):
    # explainer: n=None returns all terms
    words, scores, names = terms(kind)
    row = scores[names.index(parcel)]
    order = np.argsort(-row)[:n] if n else np.argsort(-row)
    return [(words[i], round(float(row[i]), 2)) for i in order if round(float(row[i]), 2) >= 0.01]


def for_tumor(overlap, kind="functional"):
    # decision: each involved parcel's profile weighted by how much of it is destroyed
    words, scores, names = terms(kind)
    index = {name: i for i, name in enumerate(names)}

    weights = np.zeros(len(names), dtype=np.float32)
    for row in overlap["parcels"]:
        if row["parcel"] in index:
            weights[index[row["parcel"]]] = row["fraction"]
    if weights.sum() == 0:
        return []

    # decision: every term scores something, so cut on the score not a rank
    total = (scores * weights[:, None]).sum(0) / weights.sum()
    return [{"term": words[i], "score": round(float(total[i]), 2)}
            for i in np.argsort(-total) if round(float(total[i]), 2) >= 0.01]


def drivers(term, overlap, n=None, kind="functional"):
    words, scores, names = terms(kind)
    if term not in words:
        return []
    column = words.index(term)
    index = {name: i for i, name in enumerate(names)}

    rows = [(r["parcel"], r["fraction"]) for r in overlap["parcels"] if r["parcel"] in index]
    total = sum(f for _, f in rows)
    if total == 0:
        return []

    out = [(name, fraction, float(scores[index[name], column]) * fraction / total)
           for name, fraction in rows]
    out.sort(key=lambda t: -t[2])
    return [(name, fraction, round(share, 3)) for name, fraction, share in out[:n] if share > 0]
