# One speed per voxel: every constraint is a multiplier on the same field.

import numpy as np
from scipy import ndimage

from plan import barriers, conduits

# why: white matter carries glioma roughly five times as fast as gray [swanson00]
WM_GAIN = 5.0
TRACT_GAIN = 3.0
RISK_GAIN = 1.0
SVZ_GAIN = 3.0
# why: a 6 MV penumbra is 3-5 mm, so finer features are not delivered [penumbra]
SIGMA_MM = 5.0
# a corridor thinner than 1.2 sigma is erased by the smoothing that follows
TUBE_MM = 6.0
# why: 90-95% of recurrences fall within 2 cm of the margin [recur21]
CEILING_MM = 20.0
FLOOR_MM = 5.0
# CSF has nothing to invade; nonzero only so the solver can divide by it
EPS_CSF = 0.05
SATELLITE_ML = 2.0


def field(subject, wm_gain=WM_GAIN, tract_gain=TRACT_GAIN, tube_mm=TUBE_MM,
          risk_gain=RISK_GAIN, sigma_mm=SIGMA_MM, eps_csf=EPS_CSF, svz=False,
          svz_gain=SVZ_GAIN, spare=()):
    # Everything the plan asks for is a multiplier on one field.
    if tube_mm < 1.2 * sigma_mm:
        raise ValueError(f"tube_mm {tube_mm} is erased by sigma {sigma_mm}; "
                         f"needs at least {1.2 * sigma_mm}")
    m = barriers.masks(subject)

    base = np.zeros(m["brain"].shape, dtype=np.float32)
    base[m["brain"]] = 1.0
    base[m["wm"]] = wm_gain
    base[m["csf"] | m["ventricle"]] = eps_csf
    base[m["falx"] | m["tentorium"]] = 0.0
    base[~m["brain"]] = 0.0

    near = conduits.corridors(subject, tube_mm) & m["brain"]
    reach = ndimage.distance_transform_edt(~near)
    base *= 1.0 + (tract_gain - 1.0) * np.exp(-(reach / tube_mm) ** 2)

    base *= 1.0 + risk_gain * conduits.disruption(subject)

    if svz:
        left, right = barriers.svz(barriers.labels(subject))
        touch = barriers.svz_contact(subject)
        for side, rind in zip(touch, (left, right)):
            if side:
                base[rind & m["brain"]] *= svz_gain

    for mask, weight in spare:
        base[mask] /= 1.0 + weight

    # travel time is distance over speed, so the ceiling only means millimeters
    # if the fastest tissue runs at 1; scaling every speed leaves the order alone
    peak = float(base.max())
    return base / peak if peak > 0 else base
