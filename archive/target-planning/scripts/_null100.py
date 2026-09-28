import statistics as st
import numpy as np
from plan import planeval, target, baseline
from plan.measure import _ml, _seg

SUBS = ("00055", "00285", "00352", "00129", "00307")
DRAWS = 100
print(f"corridor null, {DRAWS} draws, tube 2 mm, budget rejects 45% of reachable")
print(f"{'sub':>7}{'budget':>8}{'reject':>8}{'real':>9}{'null mean':>11}{'sd':>8}"
      f"{'gap':>9}{'z':>7}{'beat':>9}{'p':>8}", flush=True)
ranks = []
for sub in SUBS:
    f, l = f"UPENN-GBM-{sub}_11", f"UPENN-GBM-{sub}_21"
    ml = _ml(_seg(f)[1])
    reach = (baseline.expand(f, 20.0)
             & np.isfinite(target.risk(f, tube_mm=2.0))).sum() * ml
    want = round(reach * 0.55)
    real, _ = target.build(f, want, tube_mm=2.0)
    rc = planeval.coverage(l, real)["coverage"]
    cov = []
    for seed in range(1, DRAWS + 1):
        with planeval._tracts(planeval.elsewhere(f, seed)):
            m, _ = target.build(f, want, tube_mm=2.0)
        cov.append(planeval.coverage(l, m)["coverage"])
    mu, sd = st.mean(cov), st.stdev(cov)
    beat = sum(1 for c in cov if rc > c)
    p = (DRAWS - beat + 1) / (DRAWS + 1)
    ranks.append(p)
    z = (rc - mu) / sd if sd > 1e-12 else float("nan")
    print(f"{sub:>7}{want:>8.0f}{reach-want:>8.0f}{rc:>9.4f}{mu:>11.4f}{sd:>8.4f}"
          f"{rc-mu:>+9.4f}{z:>7.1f}{beat:>6}/{DRAWS}{p:>8.3f}", flush=True)

chi = -2 * sum(np.log(max(p, 1e-12)) for p in ranks)
print(f"\nFisher combined over {len(ranks)} subjects: chi2={chi:.1f}, df={2*len(ranks)}")
from scipy import stats
print(f"combined p = {stats.chi2.sf(chi, 2*len(ranks)):.2e}")
