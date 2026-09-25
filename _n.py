import statistics as st
from plan import planeval, target, baseline
SUBS = ("00055", "00285", "00352")
print("corridor null at the full 2 cm budget, tube 2 mm, Euclidean cap")
print(f"{'sub':>7}{'budget':>9}{'real':>9}{'random mean':>13}{'sd':>8}{'gap':>9}{'z':>7}{'beat':>7}")
for sub in SUBS:
    f, l = f"UPENN-GBM-{sub}_11", f"UPENN-GBM-{sub}_21"
    want = baseline.volume(f, baseline.rtog(f))
    real, _ = target.build(f, want, tube_mm=2.0)
    rc = planeval.coverage(l, real)["coverage"]
    cov = []
    for seed in range(1, 7):
        with planeval._tracts(planeval.elsewhere(f, seed)):
            m, _ = target.build(f, want, tube_mm=2.0)
        cov.append(planeval.coverage(l, m)["coverage"])
    mu, sd = st.mean(cov), st.stdev(cov)
    z = (rc - mu) / sd if sd else float("nan")
    print(f"{sub:>7}{want:>9.0f}{rc:>9.4f}{mu:>13.4f}{sd:>8.4f}"
          f"{rc-mu:>+9.4f}{z:>7.1f}{sum(1 for c in cov if rc > c):>4}/6")
