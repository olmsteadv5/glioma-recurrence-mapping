#!/home/vilem/Projects/YBAParcelVisualizer/.venv/bin/python
# Read outputs/results.json. Run whole, or cell by cell with Zed's REPL.

# %% load
import json

from paths import RESULTS

results = json.load(open(RESULTS))
subject = sorted(results)[0]
print(f"{len(results)} subject(s): {', '.join(sorted(results))}")

# %% summary table
print(f"{'subject':<22}{'WT':>7}{'TC':>7}{'ET':>7}   {'Dice WT/TC/ET':<24}{'in parcels':>22}")
for name in sorted(results):
    d = results[name]
    g, a, dice = d["regions_ml"], d.get("atlas", {}), d.get("dice")
    scores = "  ".join(f"{dice[k]:.3f}" for k in ("WT", "TC", "ET")) if dice else "-"
    spread = f"{a['in_parcels_ml']:.1f} of {a['tumour_ml']:.1f} mL, {len(a['parcels'])}"
    print(f"{name:<22}{g['WT']:>7.1f}{g['TC']:>7.1f}{g['ET']:>7.1f}   {scores:<24}{spread:>22}")

# %% parcels for one subject
for row in results[subject]["atlas"]["parcels"][:15]:
    print(f"  {row['fraction']:>6.1%}  {row['long_name']:<44} {row['lobe']:<12} {row['tissue']}")

# %% where the white matter leads
for row in results[subject]["connectome"]["targets"][:15]:
    print(f"  {row['strength']:>7.3f}  {row['long_name']:<44} {row['lobe']:<12} "
          f"{row['mean_length_mm']:>6.1f} mm")

# %% 3D view
# import view
# view.run(subject, results[subject]["atlas"])
