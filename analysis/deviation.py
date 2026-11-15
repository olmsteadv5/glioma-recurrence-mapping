#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: does diffusion that is abnormal for this patient, in either direction, mark where the tumor later recurs

import csv
import functools
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from paths import results_dir, out_dir
from show.render import follow_up_on_first

MAPS = ("md", "dti_fa", "fw", "tissue_md", "tissue_fa")
# explainer: fernet's tissue values are unstable where a voxel is nearly all fluid
TISSUE_MAPS = ("tissue_md", "tissue_fa")
TISSUES = {"white matter": (2, 41), "cortex": (3, 42), "deep gray": (10, 11, 12, 13, 49, 50, 51, 52)}
# decision: healthy reference tissue this far from the tumor, checked at 20 and 40
REFERENCE_MM = 30
# decision: the analysed band around the tumor, in 4 mm steps since the diffusion data is about 1.7 x 1.7 x 3 mm
REACH_MM, BAND_MM = 40, 4
# decision: free water above this marks a voxel too fluid for its tissue values, checked at 0.7 and 0.9
FW_MAX = 0.8
# decision: about the 3 mm neighbourhood the mirror point is averaged over, it absorbs a few mm of mirror error
MIRROR_SIGMA_MM = 1.5
# decision: stable voxels kept at this rate with matching weight, recurring voxels all kept
STABLE_RATE = 0.2
# decision: the models fit on an even share of the table, weights carried, small enough to fit in memory beside other jobs
MODEL_RATE = 0.1
OUTCOMES = {0: "stable", 1: "enhancing", 2: "non-enhancing"}


def _load(path):
    return np.asarray(nib.load(path).dataobj)


def _mirror(first, image, shape):
    # explainer: each voxel's twin across the patient's own curved midline, the lateral axis is the first one on these grids
    sheet = np.load(out_dir(first) / "midline" / "sheet.npz")["sheet"]
    inverse = np.linalg.inv(image.affine)
    x = image.affine[0, 0] * np.arange(shape[0]) + image.affine[0, 3]
    twin = 2 * sheet[None, :, :] - x[:, None, None]
    i = np.rint(inverse[0, 0] * twin + inverse[0, 3]).astype(int)
    inside = (i >= 0) & (i < shape[0])
    return np.clip(i, 0, shape[0] - 1), inside


def _smooth(values, keep, image):
    # explainer: masked gaussian mean, healthy voxels only, so tumor or missing data on the far side never leaks in
    sigma = MIRROR_SIGMA_MM / np.array(image.header.get_zooms()[:3])
    weight = ndimage.gaussian_filter(keep.astype(np.float32), sigma)
    total = ndimage.gaussian_filter(np.where(keep, values, 0).astype(np.float32), sigma)
    return np.where(weight > 0.2, total / np.maximum(weight, 1e-6), np.nan)


def patient(first, reference_mm=REFERENCE_MM, fw_max=FW_MAX):
    later = first[:-2] + "21"
    w = out_dir(first) / "diffusion_prep"
    image = nib.load(out_dir(first) / "seg.nii.gz")
    seg = _load(out_dir(first) / "seg.nii.gz")
    shape = seg.shape
    brain = _load(out_dir(first) / "brainmask.nii.gz") > 0
    tissue = _load(out_dir(first) / "tissue.nii.gz")
    seen = _load(w / "coverage_on_t2.nii.gz") >= 0.5
    after = np.asarray(follow_up_on_first(first, later).dataobj)
    whole, core = seg > 0, np.isin(seg, (1, 3))
    zooms = image.header.get_zooms()[:3]
    to_whole = ndimage.distance_transform_edt(~whole, sampling=zooms)
    to_core = ndimage.distance_transform_edt(~core, sampling=zooms) if core.any() else to_whole
    kind = np.zeros(shape, np.int8)
    for k, (_, labels) in enumerate(TISSUES.items(), start=1):
        kind[np.isin(tissue, labels)] = k
    fluid = ndimage.binary_dilation(np.nan_to_num(_load(w / "fw_on_t2.nii.gz")) > fw_max)
    twin, inside = _mirror(first, image, shape)
    _, jj, kk = np.indices(shape)
    mirror_kind, mirror_whole = kind[twin, jj, kk], whole[twin, jj, kk]
    healthy = brain & seen & ~whole & (to_whole >= reference_mm)

    # decision: candidates are what could become new tumor, existing enhancing tumor and necrosis are excluded, edema stays as its own compartment, a later cavity is neither tumor nor tissue
    candidate = brain & seen & (kind > 0) & ~np.isin(seg, (1, 3)) & (to_whole <= REACH_MM) & (after != 4)
    edema = seg == 2
    band = np.where(edema, to_core, to_whole)
    # decision: in edema only new non-enhancing core counts as non-enhancing recurrence, edema that is still edema is persistence not recurrence
    outcome = np.where(after == 3, 1, np.where((after == 1) | ((after == 2) & ~edema), 2, 0))
    rng = np.random.default_rng(int(first[-8:-3]))
    keep = candidate & ((outcome > 0) | (rng.random(shape) < STABLE_RATE))
    idx = np.nonzero(keep)
    out = {"patient": np.full(idx[0].size, int(first[-8:-3]), np.int32),
           "compartment": edema[idx].astype(np.int8),
           "band": np.minimum(band[idx] // BAND_MM, REACH_MM // BAND_MM - 1).astype(np.int8),
           "distance": band[idx].astype(np.float32),
           "tissue": kind[idx],
           "outcome": outcome[idx].astype(np.int8),
           "weight": np.where(outcome[idx] > 0, 1.0, 1 / STABLE_RATE).astype(np.float32)}
    for name in MAPS:
        values = _load(w / f"{name}_on_t2.nii.gz").astype(np.float32)
        valid = np.isfinite(values) & ~(fluid if name in TISSUE_MAPS else np.zeros(shape, bool))
        z = np.full(shape, np.nan, np.float32)
        for k in range(1, len(TISSUES) + 1):
            ref = values[healthy & valid & (kind == k)]
            if ref.size < 200:
                continue
            centre, spread = np.median(ref), 1.4826 * np.median(np.abs(ref - np.median(ref)))
            here = kind == k
            z[here] = (values[here] - centre) / max(spread, 1e-6)
        twin_ok = inside & valid & (mirror_kind == kind) & ~mirror_whole
        smoothed = _smooth(values, valid & ~whole & brain, image)
        # explainer: a twin of zero, like free water in solid white matter, gives no usable ratio
        twin_value = np.where(twin_ok & (smoothed[twin, jj, kk] > 0), smoothed[twin, jj, kk], np.nan)
        z = np.where(valid, z, np.nan)
        out[f"{name}_z"] = z[idx]
        out[f"{name}_ratio"] = np.where(valid, values / twin_value, np.nan)[idx].astype(np.float32)
        out[f"{name}_value"] = np.where(valid, values, np.nan)[idx]
    return out


def sanity(first):
    # explainer: far healthy white matter on the tumor's side scored against the other hemisphere, and against its own mirror twin
    rows = []
    image = nib.load(out_dir(first) / "seg.nii.gz")
    seg = _load(out_dir(first) / "seg.nii.gz")
    brain = _load(out_dir(first) / "brainmask.nii.gz") > 0
    tissue = _load(out_dir(first) / "tissue.nii.gz")
    side = _load(out_dir(first) / "midline" / "lateral.nii.gz")
    seen = _load(out_dir(first) / "diffusion_prep" / "coverage_on_t2.nii.gz") >= 0.5
    to_whole = ndimage.distance_transform_edt(seg == 0, sampling=image.header.get_zooms()[:3])
    wm = np.isin(tissue, TISSUES["white matter"]) & brain & seen & (seg == 0)
    far = wm & (to_whole >= REFERENCE_MM)
    home = side < 0 if ((seg > 0) & (side < 0)).sum() >= ((seg > 0) & (side > 0)).sum() else side > 0
    twin, inside = _mirror(first, image, seg.shape)
    _, jj, kk = np.indices(seg.shape)
    for name in MAPS:
        values = _load(out_dir(first) / "diffusion_prep" / f"{name}_on_t2.nii.gz")
        ok = far & np.isfinite(values)
        ref, test = values[ok & ~home], values[ok & home]
        if ref.size < 200 or test.size < 200:
            continue
        centre, spread = np.median(ref), 1.4826 * np.median(np.abs(ref - np.median(ref)))
        z = (test - centre) / max(spread, 1e-6)
        smoothed = _smooth(values, np.isfinite(values) & brain & (seg == 0), image)
        pair = ok & inside & far[twin, jj, kk]
        ratio = values[pair] / smoothed[twin, jj, kk][pair]
        ratio = ratio[np.isfinite(ratio)]
        rows.append({"patient": first[-8:-3], "map": name,
                     "tumor side z median": round(float(np.median(z)), 3),
                     "tumor side z spread": round(float(1.4826 * np.median(np.abs(z - np.median(z)))), 3),
                     "mirror ratio median": round(float(np.median(ratio)), 3) if ratio.size else None,
                     "mirror ratio spread": round(float(1.4826 * np.median(np.abs(ratio - np.median(ratio)))), 3) if ratio.size else None})
    return rows


def _auc(score, positive, weight):
    # explainer: weighted Mann-Whitney, the chance a recurring voxel scores higher than a stable one
    order = np.argsort(score, kind="stable")
    s, p, wt = score[order], positive[order], weight[order]
    neg = np.cumsum(np.where(p, 0, wt))
    ties = np.searchsorted(s, s, side="left")
    below = np.where(ties > 0, neg[np.maximum(ties - 1, 0)], 0)
    same = np.searchsorted(s, s, side="right") - 1
    tied = neg[same] - below
    total_pos, total_neg = wt[p].sum(), wt[~p].sum()
    if total_pos == 0 or total_neg == 0:
        return np.nan, 0.0
    return float(((below + tied / 2) * np.where(p, wt, 0)).sum() / (total_pos * total_neg)), float(total_pos * total_neg)


def cells(table, rows):
    # decision: recurring and stable voxels compared within one patient, band, tissue and compartment, so neither distance nor the patient can pose as a diffusion signal
    keys = [table[k][rows] for k in ("compartment", "tissue", "band", "patient")]
    rows = rows[np.lexsort(keys)]
    keys = np.stack([table[k][rows] for k in ("patient", "band", "tissue", "compartment")], 1)
    cut = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, 1)) + 1
    return np.split(rows, cut), keys[np.r_[0, cut], 0]


def stratified_auc(table, score, positive, groups, owners, draws=200):
    # explainer: one auc and pair count per cell, pooled by pair count, patients resampled by reweighting their cells
    auc, pairs = np.zeros(len(groups)), np.zeros(len(groups))
    for i, part in enumerate(groups):
        part = part[np.isfinite(score[part])]
        auc[i], pairs[i] = _auc(score[part], positive[part], table["weight"][part]) if part.size else (np.nan, 0.0)
    usable = (pairs > 0) & np.isfinite(auc)
    auc, pairs, owners = auc[usable], pairs[usable], owners[usable]
    if not pairs.sum():
        return np.nan, np.nan, np.nan
    patients, which = np.unique(owners, return_inverse=True)
    rng = np.random.default_rng(0)
    spread = []
    for _ in range(draws):
        times = np.bincount(rng.integers(0, patients.size, patients.size), minlength=patients.size)[which]
        spread.append((auc * pairs * times).sum() / max((pairs * times).sum(), 1e-12))
    low, high = np.percentile(spread, [2.5, 97.5])
    return (auc * pairs).sum() / pairs.sum(), low, high


def aucs(table):
    rows_out = []
    for outcome in (1, 2):
        positive = table["outcome"] == outcome
        for compartment, label in ((1, "edema"), (0, "normal-appearing")):
            rows = np.nonzero((table["compartment"] == compartment) & np.isin(table["outcome"], (0, outcome)))[0]
            groups, owners = cells(table, rows)
            for name in MAPS:
                for score_name, score in ((f"{name} z", table[f"{name}_z"]), (f"{name} |z|", np.abs(table[f"{name}_z"])),
                                          (f"{name} ratio to mirror", table[f"{name}_ratio"])):
                    auc, low, high = stratified_auc(table, score, positive, groups, owners)
                    rows_out.append({"outcome": OUTCOMES[outcome], "compartment": label, "score": score_name,
                                     "auc": round(auc, 3), "ci_low": round(low, 3), "ci_high": round(high, 3),
                                     "recurring_voxels": int((positive[rows]).sum()),
                                     "patients": int(np.unique(table["patient"][rows][positive[rows]]).size)})
    return rows_out


def models(table):
    # explainer: does diffusion add to distance alone, judged on patients the model never saw, with patients as clusters for the effect sizes
    import statsmodels.api as sm
    out = []
    # decision: distance gets its own slope inside edema, where it is measured to the core instead of the whole tumor
    base = ["distance", "edema", "distance x edema"]
    data = {"distance": table["distance"], "edema": table["compartment"].astype(float)}
    data["distance x edema"] = data["distance"] * data["edema"]
    for name in ("md", "fw", "dti_fa", "tissue_md"):
        data[f"{name}_z"] = np.clip(table[f"{name}_z"], -10, 10)
        data[f"{name}_absz"] = np.abs(data[f"{name}_z"])
    # decision: the main set leaves fernet's tissue md out, it is missing in the most fluid edema and would drop those voxels from every model
    sets = {"diffusion": [f"{n}_{s}" for n in ("md", "fw", "dti_fa") for s in ("z", "absz")],
            "diffusion + tissue md": [f"{n}_{s}" for n in ("md", "fw", "dti_fa", "tissue_md") for s in ("z", "absz")]}
    sample = np.random.default_rng(1).random(table["outcome"].size) < MODEL_RATE
    for outcome, (set_name, extra) in [(o, item) for o in (1, 2) for item in sets.items()]:
        rows = np.nonzero(sample & np.isin(table["outcome"], (0, outcome)) & np.all([np.isfinite(data[c]) for c in extra], 0))[0]
        y = (table["outcome"][rows] == outcome).astype(float)
        weight, groups = table["weight"][rows] / MODEL_RATE, table["patient"][rows]
        for label, columns in (("distance only", base), (f"distance + {set_name}", base + extra)):
            X = sm.add_constant(np.column_stack([data[c][rows] for c in columns]))
            held = np.zeros(rows.size)
            for p in np.unique(groups):
                train, test = groups != p, groups == p
                fit = sm.GLM(y[train], X[train], family=sm.families.Binomial(), freq_weights=weight[train]).fit()
                held[test] = fit.predict(X[test])
            # decision: scored within each patient and pooled by pair count, a pooled ranking would mostly reflect how much each patient recurs
            scored = [_auc(held[groups == p], y[groups == p] > 0, weight[groups == p]) for p in np.unique(groups)]
            auc = sum(a * n for a, n in scored if n and np.isfinite(a)) / max(sum(n for a, n in scored if n and np.isfinite(a)), 1e-12)
            # decision: a weighted logistic fit with errors clustered by patient, a GEE builds per-patient matrices too large for tens of thousands of voxels each
            gee = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weight).fit(cov_type="cluster", cov_kwds={"groups": groups})
            row = {"outcome": OUTCOMES[outcome], "model": label, "held_out_auc": round(auc, 3), "patients": int(np.unique(groups).size)}
            for c, b, pval in zip(["intercept"] + columns, gee.params, gee.pvalues):
                row[f"{c}_coef"], row[f"{c}_p"] = round(float(b), 4), float(f"{pval:.3g}")
            out.append(row)
    return out


def _write(rows, name):
    path = results_dir("native") / name
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _clean():
    # explainer: pre-op scans the diffusion quality table flags, left out of the clean sensitivity run
    with open(results_dir("native") / "diffusion_qc.csv") as fh:
        return {int(r["scan"][10:15]) for r in csv.DictReader(fh) if r["scan"].endswith("_11") and r["flags"]}


def main(cohort="upenn", step="all", reference_mm=REFERENCE_MM, fw_max=FW_MAX):
    from concurrent.futures import ProcessPoolExecutor
    reference_mm, fw_max = float(reference_mm), float(fw_max)
    # decision: sensitivity runs get their own files, the default run keeps the plain names
    tag = cohort if (reference_mm, fw_max) == (REFERENCE_MM, FW_MAX) else f"{cohort}_ref{reference_mm:g}_fw{fw_max:g}"
    saved = results_dir("native") / f"deviation_voxels_{tag}.npz"
    if step in ("models", "clean"):
        with np.load(saved) as z:
            table = {k: z[k] for k in z}
        if step == "models":
            # explainer: refit only the models from the saved voxel table
            print(_write(models(table), f"deviation_model_{tag}.csv"), flush=True)
            return
        keep = ~np.isin(table["patient"], list(_clean()))
        table = {k: v[keep] for k, v in table.items()}
        print(np.unique(table["patient"]).size, "clean patients", flush=True)
        print(_write(aucs(table), f"deviation_auc_{tag}_clean.csv"), flush=True)
        print(_write(models(table), f"deviation_model_{tag}_clean.csv"), flush=True)
        return
    firsts = [a for a, _ in measure.pairs("native") if (out_dir(a) / "diffusion_prep" / "coverage_on_t2.nii.gz").exists()
              and (out_dir(a) / "midline" / "sheet.npz").exists()]
    # decision: two at once, each patient holds several full volumes and other jobs share the laptop
    with ProcessPoolExecutor(2) as pool:
        parts = list(pool.map(functools.partial(patient, reference_mm=reference_mm, fw_max=fw_max), firsts))
        checks = [] if step == "auc" else [r for rows in pool.map(sanity, firsts) for r in rows]
    table = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(saved, **table)
    print(len(firsts), "patients,", table["outcome"].size, "voxels kept", flush=True)
    if checks:
        print(_write(checks, f"deviation_sanity_{tag}.csv"), flush=True)
    print(_write(aucs(table), f"deviation_auc_{tag}.csv"), flush=True)
    if step != "auc":
        print(_write(models(table), f"deviation_model_{tag}.csv"), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
