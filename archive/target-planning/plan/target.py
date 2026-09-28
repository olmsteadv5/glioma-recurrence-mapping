# A speed field becomes a target: solve, threshold to a budget, make it deliverable.

import functools
import subprocess

import nibabel as nib
import numpy as np
from scipy import ndimage

from plan import barriers, march, speed
from plan.baseline import expand, volume
from plan.measure import _ml, _seg
from paths import THREADS, modality, out_dir, stripped_dir
from plan.speed import (CEILING_MM, FLOOR_MM, SATELLITE_ML, SIGMA_MM, field)


def risk(subject, ceiling_mm=CEILING_MM, **params):
    tum = np.asarray(_seg(subject)[0]) > 0
    # no narrow band: it is measured in travel time, and the cap below is
    # measured in millimeters, so the two no longer mean the same thing
    time = march.travel(field(subject, **params), tum)

    # a wall voxel is the end of a road, not a road: reachable from the side the
    # front arrived on, never passed through. The sheets are two voxels thick and
    # run along midline tumors, so without this a recurrence lying against the
    # falx is unreachable, when a plan clipped at the falx would still dose it
    m = barriers.masks(subject)
    sheet = m["falx"] | m["tentorium"]
    bank = ndimage.minimum_filter(np.where(sheet, np.inf, time), size=3)
    edge = sheet & ~np.isfinite(time) & np.isfinite(bank)
    time[edge] = bank[edge] + 1.0
    return time


@functools.lru_cache(maxsize=64)
def reachable(subject, ceiling_mm=CEILING_MM):
    # the most the target could ever take: inside the cap and not behind a
    # barrier. Every gain scales a positive speed, so which voxels end up with a
    # finite arrival time is fixed by the barriers alone and does not move when
    # a parameter does. That makes this a property of the subject, and the only
    # honest denominator for "how much is the target being asked to reject".
    tum = np.asarray(_seg(subject)[0]) > 0
    return volume(subject, (expand(subject, ceiling_mm)
                            & np.isfinite(risk(subject))) | tum)


def shape(subject, budget_ml, travel_time, floor_mm=FLOOR_MM,
          ceiling_mm=CEILING_MM, sigma_mm=SIGMA_MM, satellite_ml=SATELLITE_ML):
    # From a travel-time field to something a linac can actually deliver.
    data, image = _seg(subject)
    ml = _ml(image)
    tum = data > 0

    # the floor is not negotiable, so it is spent first rather than added on top;
    # added afterwards it is un-steered Euclidean volume the budget never sees
    # the floor is Euclidean and knows nothing, so without this it buys tissue
    # the speed field priced at zero. "Microscopic disease abuts the margin"
    # does not extend to the far side of the falx.
    core = (tum | expand(subject, floor_mm)) & (np.isfinite(travel_time) | tum)

    # The cap is a distance, not a travel time. Travel time rescales whenever a
    # gain moves, so a cap measured in it silently changes how much territory is
    # reachable; the recurrence literature the 20 mm comes from measures
    # millimeters. Distance bounds the reach, a finite arrival time keeps
    # barriers honest, and travel time still decides the order.
    allowed = expand(subject, ceiling_mm) & np.isfinite(travel_time)
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


def build(subject, budget_ml, tol=0.01, travel_time=None, **params):
    take = {k: params.pop(k) for k in ("floor_mm", "satellite_ml") if k in params}
    ceiling = params.get("ceiling_mm", CEILING_MM)
    sigma = params.get("sigma_mm", SIGMA_MM)
    # a search over budgets reuses one solve; the field does not depend on them
    travel_time = risk(subject, **params) if travel_time is None else travel_time

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


def to_native(subject, mask, name="target"):
    # The atlas inflates a lesion about 13% and a whole brain about 21%, so a
    # volume only means something once it is back on the patient's own grid.
    from show import compare

    reference = modality(stripped_dir(subject), subject, "T1")
    if not reference.exists():
        raise RuntimeError(f"{subject}: no stripped T1 to resample onto")

    source = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")
    held = out_dir(subject) / f"{name}_mni.nii.gz"
    dest = out_dir(subject) / f"{name}_native.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), source.affine), held)
    subprocess.run(
        ["greedy", "-d", "3", "-rf", str(reference), "-ri", "LABEL", "0.2vox",
         "-rm", str(held), str(dest), "-r",
         str(out_dir(subject) / "affine.mat") + ",-1", str(compare._inverse(subject)),
         "-threads", THREADS], capture_output=True)
    held.unlink(missing_ok=True)
    if not dest.exists():
        raise RuntimeError(f"{subject}: greedy wrote no native {name}")

    image = nib.load(dest)
    native = np.asarray(image.dataobj) > 0
    ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0
    return native, round(float(native.sum()) * ml, 2)
