# A speed field becomes a target: solve, threshold to a budget, make it deliverable.

import numpy as np
from scipy import ndimage

from plan import barriers, march, speed
from plan.baseline import expand, volume
from plan.measure import _ml, _seg
from plan.speed import (CEILING_MM, FLOOR_MM, SATELLITE_ML, SIGMA_MM, field)


def risk(subject, ceiling_mm=CEILING_MM, **params):
    # Travel time from the tumor, marched only as far as the ceiling.
    tum = np.asarray(_seg(subject)[0]) > 0
    return march.travel(field(subject, **params), tum, narrow=ceiling_mm)


def shape(subject, budget_ml, travel_time, floor_mm=FLOOR_MM,
          ceiling_mm=CEILING_MM, sigma_mm=SIGMA_MM, satellite_ml=SATELLITE_ML):
    # From a travel-time field to something a linac can actually deliver.
    data, image = _seg(subject)
    ml = _ml(image)
    brain = barriers.masks(subject)["brain"]
    tum = data > 0

    # the floor is not negotiable, so it is spent first rather than added on top;
    # added afterwards it is un-steered Euclidean volume the budget never sees
    core = tum | expand(subject, floor_mm)

    # the budget fills what is left, nearest first in travel order
    allowed = brain & (travel_time <= ceiling_mm)
    want = max(int(budget_ml / ml) - int(core.sum()), 0)
    spare_room = allowed & ~core
    times = travel_time[spare_room]
    if want < times.size:
        cut = np.partition(times, want)[want] if want else -np.inf
        mask = core | (spare_room & (travel_time < cut))
    else:
        mask = core | spare_room

    # components BEFORE smoothing, while a thin corridor still connects them
    tagged, count = ndimage.label(mask)
    keep = np.zeros_like(mask)
    holds = set(np.unique(tagged[tum])) - {0}
    satellites = []
    for n in range(1, count + 1):
        piece = tagged == n
        size = float(piece.sum()) * ml
        if n in holds:
            keep |= piece
        elif size >= satellite_ml:
            keep |= piece
            satellites.append(round(size, 2))

    keep = ndimage.binary_closing(keep, iterations=2)
    # why: below the penumbra the dose distribution cannot follow the shape [penumbra]
    blurred = ndimage.gaussian_filter(keep.astype(np.float32), sigma_mm)
    return (blurred >= 0.5) | tum, satellites


def build(subject, budget_ml, tol=0.01, **params):
    take = {k: params.pop(k) for k in ("floor_mm", "satellite_ml") if k in params}
    ceiling = params.get("ceiling_mm", CEILING_MM)
    sigma = params.get("sigma_mm", SIGMA_MM)
    travel_time = risk(subject, **params)

    # closing and smoothing move the volume, so ask for less until it lands.
    # the ceiling can make the budget unreachable, and then asking for more
    # changes nothing; stop rather than spending every iteration on it
    ask, mask, satellites, was, capped = budget_ml, None, [], None, False
    for _ in range(12):
        mask, satellites = shape(subject, ask, travel_time,
                                 ceiling_mm=ceiling, sigma_mm=sigma, **take)
        got = volume(subject, mask)
        if abs(got - budget_ml) <= tol * budget_ml:
            break
        if was is not None and abs(got - was) <= tol * max(got, 1.0):
            capped = True
            break
        was = got
        ask = max(ask * budget_ml / max(got, 1.0), 1.0)
    return mask, {"volume_ml": volume(subject, mask),
                  "asked_ml": round(budget_ml, 2),
                  "capped": capped,
                  "satellites_ml": satellites,
                  "reach_mm": round(float(travel_time[mask & np.isfinite(travel_time)].max()), 1)
                  if mask.any() else 0.0}
