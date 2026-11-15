#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: one row per preprocessed diffusion scan, from files the pipeline already wrote, with flags for scans worth a second look

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import out_dir, results_dir
from scan.diffusion import PREP
from show.render import follow_up_on_first

WM, VENTRICLES = (2, 41), (4, 43)
# decision: hard limits worth a look, the synb0 ones match the gate in preprocess and tiny fernet references make its fit unreliable
LIMITS = {"topup_mm": 5.0, "topup_rad": 0.1, "eddy_mean_mm": 3.0, "eddy_step_mm": 1.0, "volumes_lost": 4, "tracts_wm": 0.6,
          "fernet_wm_voxels": 500, "fernet_csf_voxels": 200,
          "tumor_covered": 0.9, "zone_covered": 0.9, "new_recurrence_covered": 0.9}
# explainer: these limits are floors, every other one is a ceiling
FLOORS = ("tracts_wm", "fernet_wm_voxels", "fernet_csf_voxels", "tumor_covered", "zone_covered", "new_recurrence_covered")
# decision: the band around the tumor the recurrence analyses test, so the part of it the diffusion scan misses matters
ZONE_MM = 40
# decision: values this many robust spreads from the cohort get a look, not a verdict
SPREAD = 3.0
COHORT = ("snr_b0", "cnr_dwi", "md_wm", "fw_wm", "fw_ventricles", "tissue_md_wm", "dti_fa_wm")
# decision: a volume with this share of its slices replaced has lost its direction more than it was patched
LOST = 0.1
# decision: every direction was scanned three times, so only more than about 5% of volumes lost gets a look


def _map(work, name):
    return np.asarray(nib.load(work / f"{name}_on_t2.nii.gz").dataobj)


def covered(subject, image, seen, brain, seg):
    # explainer: share of the tumor, the band around it and the new recurrence that the diffusion scan reached, pre-op scans only for the last two
    out = {"tumor_covered": round(float(seen[seg].mean()), 3) if seg.any() else None,
           "zone_covered": None, "new_recurrence_covered": None}
    later = subject[:-2] + "21"
    if not seg.any() or not subject.endswith("_11") or not (out_dir(later) / "seg.nii.gz").exists():
        return out
    far = ndimage.distance_transform_edt(~seg, sampling=image.header.get_zooms()[:3])
    zone = brain & ~seg & (far <= ZONE_MM)
    # decision: only recurrence that lands in the pre-op brain, the rest is resection or shift the analyses can't use either
    new = (np.asarray(follow_up_on_first(subject, later).dataobj) > 0) & brain & ~seg
    share = lambda region: round(float(seen[region].mean()), 3) if region.any() else None
    out["zone_covered"], out["new_recurrence_covered"] = share(zone), share(new)
    return out


def row(subject):
    d, w = PREP / subject, out_dir(subject) / "diffusion_prep"
    moved = np.loadtxt(d / "synb0_out" / "topup_movpar.txt")[1]
    rms = np.atleast_2d(np.loadtxt(d / "eddy.eddy_movement_rms"))
    # explainer: restricted leaves out the phase-encode shift, which eddy can't tell apart from its own field
    restricted = np.atleast_2d(np.loadtxt(d / "eddy.eddy_restricted_movement_rms"))
    weighted = np.loadtxt(d / "dwi.bval") > 50
    outliers = np.loadtxt(d / "eddy.eddy_outlier_map", skiprows=1)[weighted]
    # explainer: eddy's first cnr volume is the b0 snr, the second the diffusion-weighted cnr, both on the scan's own grid
    cnr = np.asarray(nib.load(d / "eddy.eddy_cnr_maps.nii.gz").dataobj)
    reference = np.asarray(nib.load(d / "fernet" / "fernet_wm_mask.nii.gz").dataobj) > 0
    tissue = np.asarray(nib.load(out_dir(subject) / "tissue.nii.gz").dataobj)
    seg = np.asarray(nib.load(out_dir(subject) / "seg.nii.gz").dataobj)
    brain = np.asarray(nib.load(out_dir(subject) / "brainmask.nii.gz").dataobj) > 0
    # explainer: normal-appearing only, tumor and its labels would pull the medians
    wm, ventricles = np.isin(tissue, WM) & (seg == 0), np.isin(tissue, VENTRICLES) & (seg == 0)
    prep = json.loads((w / "check.json").read_text())
    before = out_dir(subject) / "diffusion" / "check.json"
    raw = json.loads(before.read_text()) if before.exists() else {}
    out = {"scan": subject,
           "topup_mm": round(float(np.linalg.norm(moved[:3])), 2),
           "topup_rad": round(float(np.abs(moved[3:]).max()), 4),
           "eddy_mean_mm": round(float(rms[:, 0].mean()), 2),
           "eddy_step_mm": round(float(rms[1:, 1].mean()), 2),
           "eddy_step_restricted_mm": round(float(restricted[1:, 1].mean()), 2),
           "replaced_pct": round(100 * float(outliers.mean()), 2),
           "volumes_lost": int((outliers.mean(1) > LOST).sum()),
           "snr_b0": round(float(np.median(cnr[..., 0][reference])), 2),
           "cnr_dwi": round(float(np.median(cnr[..., 1][reference])), 3),
           "fernet_wm_voxels": int((np.asarray(nib.load(d / "fernet" / "fernet_wm_mask.nii.gz").dataobj) > 0).sum()),
           "fernet_csf_voxels": int((np.asarray(nib.load(d / "fernet" / "fernet_csf_mask.nii.gz").dataobj) > 0).sum()),
           "outside_fov_pct": round(100 * float((_map(w, "coverage")[brain] < 0.5).mean()), 1)}
    image = nib.load(out_dir(subject) / "seg.nii.gz")
    out.update(covered(subject, image, _map(w, "coverage") >= 0.5, brain, seg > 0))
    for name, region in (("md", wm), ("fw", wm), ("fw", ventricles), ("tissue_md", wm), ("dti_fa", wm)):
        tag = "wm" if region is wm else "ventricles"
        out[f"{name}_{tag}"] = round(float(np.nanmedian(_map(w, name)[region])), 3)
    out.update({"tracts_wm": prep["white_matter"], "tracts_cortex": prep["cortex"], "tracts_outside": prep["outside_brain"],
                "tracts_wm_before": raw.get("white_matter"), "fa_contrast": round(prep["fa_white_matter"] - prep["fa_cortex"], 3),
                "fa_contrast_before": round(raw["fa_white_matter"] - raw["fa_cortex"], 3) if raw else None})
    return out


def flag(rows):
    # explainer: nan when a region has no covered voxels, skipped so one scan can't blank a whole column
    centre = {k: np.nanmedian([r[k] for r in rows]) for k in COHORT}
    spread = {k: 1.4826 * np.nanmedian([abs(r[k] - centre[k]) for r in rows]) for k in COHORT}
    for r in rows:
        why = [k for k, limit in LIMITS.items() if r[k] is not None and (r[k] < limit if k in FLOORS else r[k] > limit)]
        # decision: signal and contrast only count against a scan when low, a clean scan is no reason for a look
        why += [f"{k} unusual" for k in COHORT if spread[k] and (centre[k] - r[k] if k in ("snr_b0", "cnr_dwi")
                                                                  else abs(r[k] - centre[k])) > SPREAD * spread[k]]
        why += [f"{k} missing" for k in COHORT if np.isnan(r[k])]
        r["flags"] = "; ".join(why)
    return rows


def main():
    scans = sorted(p.name for p in PREP.iterdir() if (out_dir(p.name) / "diffusion_prep" / "check.json").exists())
    rows = flag([row(s) for s in scans])
    path = results_dir("native") / "diffusion_qc.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = [r for r in rows if r["flags"]]
    print(f"{len(rows)} scans, {len(flagged)} flagged, written to {path}")
    for r in flagged:
        print(f"  {r['scan']}: {r['flags']}")


if __name__ == "__main__":
    main()
