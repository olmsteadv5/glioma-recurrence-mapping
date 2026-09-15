# Structural connectivity: where the white matter from the involved parcels leads.

import csv

from atlas import ensure, parcels

# SC_PathCount is hop count, so 1 means a direct white matter connection.
DIRECT = 1


def load():
    # Direct edges only, as {parcel: [(other, strength, mean_length_mm), ...]}.
    edges = {}
    with open(ensure("structuralconnectivity_metrics_mean.csv")) as fh:
        for row in csv.DictReader(fh):
            if int(float(row["SC_PathCount"])) != DIRECT:
                continue
            strength = float(row["SC_Count"])
            if strength <= 0:
                continue
            a, b = row["Parcel_A_name"], row["Parcel_B_name"]
            length = float(row["SC_Length"])
            edges.setdefault(a, []).append((b, strength, length))
            edges.setdefault(b, []).append((a, strength, length))
    for value in edges.values():
        value.sort(key=lambda e: -e[1])
    return edges


def targets(overlap, edges, min_fraction=0.2, top=25):
    # Parcels outside the tumour that the involved parcels connect to directly.
    meta = {p["name"]: p for p in parcels()}
    involved = {r["parcel"] for r in overlap["parcels"]}
    seeds = [r["parcel"] for r in overlap["parcels"] if r["fraction"] >= min_fraction]

    reached = {}
    for seed in seeds:
        for other, strength, length in edges.get(seed, []):
            if other in involved:
                continue
            entry = reached.setdefault(other, {"strength": 0.0, "from": [], "length": length})
            entry["strength"] += strength
            entry["from"].append(seed)

    rows = []
    for name, entry in reached.items():
        info = meta.get(name, {})
        rows.append({
            "parcel": name,
            "long_name": info.get("long_name", name),
            "lobe": info.get("lobe", ""),
            "strength": round(entry["strength"], 4),
            "mean_length_mm": round(entry["length"], 1),
            "from_n_involved": len(entry["from"]),
        })

    rows.sort(key=lambda r: -r["strength"])
    return {"seed_parcels": len(seeds), "targets": rows[:top]}
