# Scoring a target against what the tumor actually did next.

import numpy as np

from plan import baseline
from plan.measure import _ml, _seg


def coverage(later, mask):
    # the share of the later tumor a target catches
    data, image = _seg(later)
    hit = data > 0
    caught = float((hit & mask).sum())
    return {"coverage": round(caught / max(float(hit.sum()), 1), 4),
            "caught_ml": round(caught * _ml(image), 2),
            "missed_ml": round(float((hit & ~mask).sum()) * _ml(image), 2)}


def curve(first, later, target, budgets):
    # coverage against volume for one method, read either way round
    out = []
    for want in budgets:
        mask = target(first, want)
        got = baseline.volume(first, mask)
        out.append((got, coverage(later, mask)["coverage"]))
    return out


def volume_for(first, later, target, want, lo=10.0, hi=900.0, tol=0.005):
    # the inverse reading: the smallest volume that reaches a coverage
    for _ in range(30):
        mid = (lo + hi) / 2
        got = coverage(later, target(first, mid))["coverage"]
        if abs(got - want) <= tol:
            break
        lo, hi = (mid, hi) if got < want else (lo, mid)
    return round(mid, 1), round(got, 4)


def sphere(first, volume_ml):
    # the volume-matched isotropic comparator, as a target callable
    return baseline.isotropic(first, volume_ml)[0]


def sweep(pairs, target, budgets):
    # one row per pair; stage 1 asks only whether anything moves at all
    rows = []
    for first, later in pairs:
        rows.append((first[-5:], curve(first, later, target, budgets)))
    return rows
