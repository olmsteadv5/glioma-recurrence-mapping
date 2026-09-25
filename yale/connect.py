# Yale structural connectivity: which intact parcels lose their input to a tumor.

import csv
import functools

import numpy as np

from yale.atlas import ATLAS_DIR, ensure, parcels

FILE = "structuralconnectivity_metrics_mean.csv"
METRIC = "SC_Count"


@functools.lru_cache(maxsize=1)
def matrix():
    # The file is an edge list of 241860 parcel pairs, not a matrix; fold it into one.
    cache = ATLAS_DIR / "connectivity.npz"
    if cache.exists():
        return np.load(cache)["weights"]

    names = [p["name"] for p in parcels()]
    index = {name: i for i, name in enumerate(names)}
    weights = np.zeros((len(names), len(names)), dtype=np.float32)
    with open(ensure(FILE)) as fh:
        rows = csv.DictReader(fh)
        for row in rows:
            a, b = index.get(row["Parcel_A_name"]), index.get(row["Parcel_B_name"])
            value = row.get(METRIC) or 0
            if a is None or b is None or not value:
                continue
            weights[a, b] = weights[b, a] = float(value)

    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, weights=weights)
    return weights


def at_risk(overlap, floor=0.01):
    # For each untouched parcel, how much of its connectivity runs into the tumor.
    names = [p["name"] for p in parcels()]
    index = {name: i for i, name in enumerate(names)}
    weights = matrix()

    hit = np.zeros(len(names), dtype=np.float32)
    for row in overlap["parcels"]:
        if row["parcel"] in index:
            hit[index[row["parcel"]]] = row["fraction"]
    if hit.sum() == 0:
        return []

    lost = weights @ hit
    total = weights.sum(1)
    share = np.divide(lost, total, out=np.zeros_like(lost), where=total > 0)

    out = [{"parcel": names[i], "share": round(float(share[i]), 4),
            "weight": round(float(lost[i]), 3)}
           for i in np.argsort(-share) if hit[i] == 0 and round(float(share[i]), 4) >= floor]
    return out


def partners(parcel, overlap, n=None):
    # The involved parcels one untouched parcel is losing, as shares of its own
    # loss: the raw edge weights are tiny and all round to nothing on their own.
    names = [p["name"] for p in parcels()]
    index = {name: i for i, name in enumerate(names)}
    if parcel not in index:
        return []
    weights = matrix()[index[parcel]]

    out = []
    for row in overlap["parcels"]:
        if row["parcel"] in index:
            edge = float(weights[index[row["parcel"]]]) * row["fraction"]
            if edge > 0:
                out.append([row["parcel"], row["fraction"], edge])
    total = sum(edge for _, _, edge in out) or 1
    for entry in out:
        entry[2] = round(entry[2] / total, 4)
    out.sort(key=lambda t: -t[2])
    return [tuple(entry) for entry in out[:n]]
