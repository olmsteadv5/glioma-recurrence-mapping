# Fast marching: a speed at every voxel in, an arrival time at every voxel out.

import numpy as np
from scipy import ndimage

_LEVEL = {}


def _level_set(seed, dx):
    # Put the zero contour on the seed's own rind, so travel time is measured
    # from the same voxel centers a distance transform measures from. Signing
    # the plain transform instead leaves the contour half a voxel out and
    # overshoots a 2 cm expansion by twice as much.
    key = (seed.shape, np.ravel(dx).tobytes(), hash(seed.tobytes()))
    if key not in _LEVEL:
        step = float(np.min(dx))
        inside = ndimage.distance_transform_edt(seed, sampling=dx)
        outside = ndimage.distance_transform_edt(~seed, sampling=dx)
        # a second solve on the same seed only ever changes the speed field
        _LEVEL.clear()
        _LEVEL[key] = np.where(seed, -(inside - step), outside)
    return _LEVEL[key]


def travel(speed, seed, dx=1.0, narrow=0.0):
    # Travel time from the seed under a variable speed, by fast marching.
    import skfmm

    speed = np.asarray(speed, dtype=np.float64)
    seed = np.ascontiguousarray(np.asarray(seed).astype(bool))
    if not seed.any():
        return np.full(speed.shape, np.inf, dtype=np.float32)
    phi = _level_set(seed, dx)

    # skfmm marches only the unmasked part, but it still builds 1/speed first
    blocked = ~(speed > 0)
    field = skfmm.travel_time(np.ma.MaskedArray(phi, blocked),
                              np.where(blocked, 1.0, speed), dx=dx, narrow=narrow)
    out = np.ma.filled(field, np.inf).astype(np.float32)
    out[~np.isfinite(out)] = np.inf
    return out
