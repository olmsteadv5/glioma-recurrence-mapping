# One speed per voxel: every constraint is a multiplier on the same field.

import numpy as np
from scipy import ndimage

from plan import barriers, conduits

# why: white matter carries glioma roughly five times as fast as gray [swanson00]
WM_GAIN = 5.0
TRACT_GAIN = 3.0
RISK_GAIN = 1.0
# at_risk shares run 0 to about a quarter, so risk_gain multiplied a number that
# was never bigger than 0.25 while wm_gain multiplied a flat 5. Dividing by a
# fixed constant makes risk_gain = 4 mean x5 at the most disrupted parcel, the
# same thing wm_gain = 5 means. Fixed on purpose: a per-subject max would make
# the parameter mean something different every time the cohort grows.
RISK_SCALE = 0.25
SVZ_GAIN = 3.0
# why: a 6 MV penumbra is 3-5 mm, so finer features are not delivered [penumbra]
# a gaussian's 80-20 edge is about 1.7 sigma, so 2.5 gives 4.2 mm. It was 5.0,
# which is an 8.4 mm edge, roughly twice any linac.
SIGMA_MM = 2.5
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
    # tube_mm steers, it is not delivered. The 1.2 x sigma floor belongs to the
    # final mask, where a lone corridor has to survive smoothing, and it was
    # wrong here: at 6 mm the tubes cover 91% of the neighborhood and stop
    # picking out any direction at all.
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

    base *= 1.0 + risk_gain * conduits.disruption(subject) / RISK_SCALE

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
