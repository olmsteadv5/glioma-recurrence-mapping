# Scoring a target against what the tumor actually did next.

import contextlib
import statistics

import nibabel as nib
import numpy as np

from paths import MNI, REGIONS, out_dir
from plan import baseline, conduits, target
from plan.speed import FLOOR_MM
from plan.measure import _ml, _seg
from yale import atlas, connect
from yale import tracts as tractography

# the share of edges the rewiring keeps. The matrix is 71% dense, so double-edge
# swaps on the raw graph reject almost everything and randomize nothing.
DENSITY = 0.1

# the usual rule for Maslov-Sneppen: enough passes that no edge is where it was
SWAPS = 10

# rtog and estro come out at whatever volume the guideline gives them; the rest
# are held to the budget and are only comparable if they reach it
MATCHED = ("isotropic", "flair", "barriers", "target", "null")
METHODS = ("rtog", "estro") + MATCHED


def coverage(later, mask, region=None):
    # the share of the later tumor a target catches; region None is the whole
    # segmentation, which every number before the endpoint change was scored on
    data, image = _seg(later)
    hit = data > 0 if region is None else np.isin(data, REGIONS[region])
    caught = float((hit & mask).sum())
    return {"coverage": round(caught / max(float(hit.sum()), 1), 4),
            "caught_ml": round(caught * _ml(image), 2),
            "missed_ml": round(float((hit & ~mask).sum()) * _ml(image), 2)}


def curve(first, later, method, budgets):
    # coverage against volume for one method, read either way round
    out = []
    for want in budgets:
        mask = method(first, want)
        got = baseline.volume(first, mask)
        out.append((got, coverage(later, mask)["coverage"]))
    return out


def volume_for(first, later, method, want, lo=10.0, hi=900.0, tol=0.005):
    # the inverse reading: the smallest volume that reaches a coverage
    for _ in range(30):
        mid = (lo + hi) / 2
        got = coverage(later, method(first, mid))["coverage"]
        if abs(got - want) <= tol:
            break
        lo, hi = (mid, hi) if got < want else (lo, mid)
    return round(mid, 1), round(got, 4)


def smallest(first, later, method, want, hi, region=None, tol=0.005, step_ml=1.0):
    # the least volume a method needs to still reach a coverage. None when even
    # its largest target falls short, which for ours means a barrier or the
    # 20 mm reach sits between the tumor and part of the recurrence
    best = method(first, hi)
    if coverage(later, best, region)["coverage"] < want - tol:
        return None
    lo = 1.0
    while hi - lo > step_ml:
        mid = (lo + hi) / 2
        mask = method(first, mid)
        if coverage(later, mask, region)["coverage"] >= want - tol:
            hi, best = mid, mask
        else:
            lo = mid
    return {"volume_ml": baseline.volume(first, best),
            "coverage": coverage(later, best, region)["coverage"]}


def spared(first, later, region=None, **params):
    # the headline endpoint: cover what the standard 2 cm plan covers, and count
    # the brain each shape needs to do it. Shrinking the uniform margin alone
    # already spares a lot, so the method only earns what it saves past that
    data, _ = _seg(later)
    if not (data > 0 if region is None else np.isin(data, REGIONS[region])).any():
        # nothing of this compartment came back, so there is nothing to cover
        return {"region": region or "all", "skipped": "absent at the later scan"}
    plan = baseline.rtog(first)
    plan_ml = baseline.volume(first, plan)
    want = coverage(later, plan, region)["coverage"]

    # the uniform expansion gets the same 5 mm minimum our target is held to.
    # Without it, a recurrence sitting inside the first outline is matched by a
    # 0 mm margin no one would treat with, and ours loses by its floor alone
    floor_ml = baseline.volume(first, baseline.expand(first, FLOOR_MM))
    iso = smallest(first, later,
                   lambda f, v: baseline.isotropic(f, max(v, floor_ml))[0],
                   want, plan_ml * 1.05, region)
    time = target.risk(first, **params)
    ours = smallest(first, later,
                    lambda f, v: target.build(f, v, travel_time=time, **params)[0],
                    want, target.reachable(first), region)
    out = {"region": region or "all", "plan_ml": plan_ml, "want": want,
           "isotropic_ml": iso and iso["volume_ml"],
           "target_ml": ours and ours["volume_ml"]}
    if iso and ours:
        out["vs_plan_ml"] = round(plan_ml - ours["volume_ml"], 1)
        out["vs_isotropic_ml"] = round(iso["volume_ml"] - ours["volume_ml"], 1)
    return out


def sphere(first, volume_ml):
    # the volume-matched isotropic comparator, as a target callable
    return baseline.isotropic(first, volume_ml)[0]


def sweep(pairs, method, budgets):
    # one row per pair; stage 1 asks only whether anything moves at all
    rows = []
    for first, later in pairs:
        rows.append((first[-5:], curve(first, later, method, budgets)))
    return rows


def rewire(weights, seed=0, density=DENSITY, swaps=SWAPS):
    # Maslov-Sneppen on the strongest edges: degree is preserved exactly, and
    # the weights are then laid back on by strength product so the strength
    # sequence survives too. swaps=0 gives the thresholded-but-intact control,
    # which is what isolates the shuffle from the thresholding.
    rng = np.random.default_rng(seed)
    upper = np.triu_indices(weights.shape[0], 1)
    flat = weights[upper]
    live = np.flatnonzero(flat > 0)
    order = live[np.argsort(-flat[live])]
    kept = order[:max(int(round(density * live.size)), 1)]
    edges = np.stack([upper[0][kept], upper[1][kept]], axis=1)

    present = {(int(a), int(b)) for a, b in edges}
    count = len(edges)
    swapped = 0
    for _ in range(swaps * count):
        first, second = rng.integers(0, count, 2)
        if first == second:
            continue
        a, b = edges[first]
        c, d = edges[second]
        if rng.random() < 0.5:
            c, d = d, c
        if len({int(a), int(b), int(c), int(d)}) < 4:
            continue
        one = (int(min(a, d)), int(max(a, d)))
        two = (int(min(c, b)), int(max(c, b)))
        if one in present or two in present or one == two:
            continue
        present.discard((int(a), int(b)))
        present.discard((int(min(c, d)), int(max(c, d))))
        present.update((one, two))
        edges[first] = one
        edges[second] = two
        swapped += 1

    strength = weights.sum(1)
    want = strength[edges[:, 0]] * strength[edges[:, 1]]
    rank = np.argsort(np.argsort(want))
    out = np.zeros_like(weights)
    out[edges[:, 0], edges[:, 1]] = np.sort(flat[kept])[rank]
    return out + out.T, swapped


def shares(subject, weights, floor=0.01):
    # at_risk's arithmetic against a matrix that is not the cached one, painted
    # onto voxels the way conduits.disruption does
    labels, names = atlas.label_volume(nib.load(MNI))
    rows = atlas.overlap(out_dir(subject) / "seg_in_mni.nii.gz",
                         labels, names)["parcels"]
    seat = {name: i for i, name in enumerate(names)}

    hit = np.zeros(len(names), dtype=np.float32)
    for row in rows:
        if row["parcel"] in seat:
            hit[seat[row["parcel"]]] = row["fraction"]

    total = weights.sum(1)
    lost = weights @ hit
    share = np.divide(lost, total, out=np.zeros_like(lost), where=total > 0)
    share = np.round(share, 4)
    share[(hit > 0) | (share < floor)] = 0.0

    painted = np.zeros(len(names) + 1, dtype=np.float32)
    painted[1:] = share
    return painted[labels]


@contextlib.contextmanager
def _instead(field):
    # conduits.disruption is lru_cached on subject. Swapping the module
    # attribute means the real function is never called, so its cache never
    # sees the shuffled field and the true one stays valid afterwards.
    real = conduits.disruption
    conduits.disruption = lambda subject: field
    try:
        yield
    finally:
        conduits.disruption = real


def null(subject, budget_ml, seed=0, density=DENSITY, swaps=SWAPS, **params):
    # the permutation test: same degrees, same strengths, wrong wiring
    weights, swapped = rewire(connect.matrix(), seed=seed, density=density,
                              swaps=swaps)
    with _instead(shares(subject, weights)):
        mask, diagnostics = target.build(subject, budget_ml, **params)
    diagnostics["swaps"] = swapped
    diagnostics["seed"] = seed
    return mask, diagnostics


@contextlib.contextmanager
def _tracts(ids):
    # corridors is not cached, so swapping the attribute is enough; crossing's
    # own cache is never consulted because the real function is not called
    real = conduits.crossing
    conduits.crossing = lambda subject: ids
    try:
        yield
    finally:
        conduits.crossing = real


def elsewhere(subject, seed=0):
    # the same number of corridors, drawn from anywhere in the atlas rather than
    # from the streamlines that actually run through the mass
    wanted = len(conduits.crossing(subject))
    total = len(tractography.centerlines())
    rng = np.random.default_rng(seed)
    return sorted(int(n) for n in rng.choice(total, size=wanted, replace=False))


def tract_null(subject, budget_ml, seed=0, **params):
    # The corridor null. rewire() permutes the connectivity matrix, which only
    # feeds disruption; the corridors come from streamline geometry and never
    # see it. This is what tests whether WHICH tracts cross the tumor matters,
    # as against merely having tubes of fast tissue somewhere.
    with _tracts(elsewhere(subject, seed)):
        return target.build(subject, budget_ml, **params)


def budget_for(first, budget):
    # a number in mL, or the name of a guideline target to match
    if budget == "estro":
        return baseline.volume(first, baseline.estro(first))
    if budget == "rtog":
        return baseline.volume(first, baseline.rtog(first))
    return float(budget)


def one(first, name, budget_ml, seed=0, **params):
    if name == "rtog":
        mask = baseline.rtog(first)
    elif name == "estro":
        mask = baseline.estro(first)
    elif name == "isotropic":
        mask = baseline.isotropic(first, budget_ml)[0]
    elif name == "flair":
        mask = baseline.flair(first, budget_ml)[0]
    elif name == "barriers":
        # barriers and CSF only: every gain flat, so nothing steers but anatomy
        return target.build(first, budget_ml, wm_gain=1.0, tract_gain=1.0,
                            risk_gain=0.0, **params)
    elif name == "target":
        return target.build(first, budget_ml, **params)
    elif name == "null":
        return null(first, budget_ml, seed=seed, **params)
    else:
        raise ValueError(f"no method named {name}")
    return mask, {"volume_ml": baseline.volume(first, mask), "capped": False}


def compare(pairs, budget="estro", methods=METHODS, seed=0, tol=0.05, **params):
    # One table, every comparator, at one volume per pair. A method that could
    # not spend the budget takes the whole pair out rather than contributing a
    # coverage read at some other volume; two rounds of results died that way.
    rows, dropped = [], []
    for first, later in pairs:
        want = budget_for(first, budget)
        row = {"pair": f"{first[-5:]} -> {later[-5:]}", "budget_ml": round(want, 2)}
        why = []
        for name in methods:
            mask, diagnostics = one(first, name, want, seed=seed, **params)
            got = diagnostics["volume_ml"]
            row[name] = {"volume_ml": got,
                         "coverage": coverage(later, mask)["coverage"]}
            off = abs(got - want) > tol * want
            if name in MATCHED and (diagnostics.get("capped") or off):
                why.append(f"{name} landed at {got} mL")
        if why:
            dropped.append({"pair": row["pair"], "budget_ml": round(want, 2),
                            "why": why})
        else:
            rows.append(row)

    median = {}
    for name in methods:
        got = [r[name]["coverage"] for r in rows if name in r]
        if got:
            median[name] = round(statistics.median(got), 4)
    return {"rows": rows, "dropped": dropped, "median": median, "n": len(rows)}
