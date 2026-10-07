#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# The per-voxel dataset (analysis/FEATURES.md): for every brain voxel of every
# pair, raw measurements from the pre-op scan and the recurrence labels from the
# follow-up, on the MNI 1 mm grid. Raw only: no rule, threshold or target from
# earlier analyses goes in; those are applied to the columns later.
#
#   ./analysis/features.py 00036          one patient, in this process
#   ./analysis/features.py --workers 3    every pair not yet done

import argparse
import csv
import json
import multiprocessing
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import measure
from analysis.walkdown import _hull
from geometry import ml_per_voxel, outline, template_brain
from hcp import streamlines as hcp
from paths import ATLAS_DIR, ATLAS_INPUT, INPUTS, MNI, OUTPUTS, PARTICIPANTS, out_dir
from yale.atlas import label_volume, parcels
from yale.connect import matrix

VERSION = 1
FEATURES = OUTPUTS / "features"
TCIA = INPUTS / "tcia" / "PKG - UPENN-GBM-NIfTI" / "UPENN-GBM" / "NIfTI-files"
ATLAS_CACHE = FEATURES / "atlas_cache.npz"
# each worker's greedy threads; three workers then use at most about half the cores
THREADS = "4"

STRUCTURAL = ("T1", "T1GD", "T2", "FLAIR")
DTI = ("FA", "TR", "AD", "RD")
DSC = ("ap-rCBV", "PH", "PSR")
# SynthSeg labels [billot23]
WM, CORTEX, SULCAL_CSF = (2, 41), (3, 42), (24,)
LATERAL_VENTRICLE, VENTRICLES = (4, 5, 43, 44), (4, 5, 14, 15, 43, 44)
CLASSES = ("association", "commissural", "projection", "cerebellum")
REFERENCES = (("wt", "WT"), ("core", "TC"))
NEAR_MM = 10        # white matter at least this far from the tumor counts as normal-appearing
WM_BOX_MM = 11      # the box the local white matter share is averaged over

# name, units, group, what it is. Columns with {ref} exist once per reference.
SPEC = [
    ("dist_{ref}", "mm", "tumor geometry", "signed straight-line distance to the edge of the pre-op {ref}, negative inside it"),
    ("hull_{ref}", "0/1", "tumor geometry", "inside the convex hull of the pre-op {ref}"),
    ("layer_{ref}", "0-1", "tumor geometry", "|direction away from the {ref} · direction of increasing depth|: 0 runs along the brain's depth layers, 1 dives across them"),
    ("curvature_{ref}", "1/mm", "tumor geometry", "mean curvature of the {ref} surface at its nearest edge point (Laplacian of the smoothed signed distance; positive bulges out)"),
    ("halo_at_core", "mm", "tumor geometry", "thickness of the FLAIR halo at the core edge point nearest the voxel: distance from that point to the outside of the whole tumor"),
    ("compartment", "label", "tumor geometry", "pre-op label: 0 outside, 1 non-enhancing core, 2 FLAIR, 3 enhancing, 4 cavity"),
    ("dist_et", "mm", "tumor geometry", "signed distance to the pre-op enhancing tumor"),
    ("dist_netc", "mm", "tumor geometry", "signed distance to the pre-op non-enhancing core"),
    ("midline", "mm", "midline", "distance from the midline plane (x = 0), positive on the side holding most of the tumor, 0 on the plane"),
    ("across_{ref}", "mm", "midline", "distance to any part of the {ref} strictly past the midline plane on the other side (voxels on the plane count as neither side); missing if none"),
    ("depth", "mm", "anatomy", "distance below the patient's own brain surface"),
    ("in_brain", "0/1", "anatomy", "inside the patient's own brain mask"),
    ("dist_ventricles", "mm", "anatomy", "distance to any ventricle"),
    ("dist_lateral", "mm", "anatomy", "distance to a lateral ventricle (its wall)"),
    ("lobe", "label", "anatomy", "lobe map value"),
    ("tissue", "label", "tissue", "SynthSeg label"),
    ("gw_border", "mm", "tissue", "distance to the gray/white border, positive in white matter"),
    ("sulcal_csf", "mm", "tissue", "distance to extracerebral CSF"),
    ("wm_share", "0-1", "tissue", f"white matter share of the {WM_BOX_MM} mm box around the voxel"),
    ("tract_dist_{ref}", "mm", "tracts", "shortest distance along any streamline through the {ref}, with no hull, stop or limit; missing if none arrives"),
    ("tract_n_{ref}", "count", "tracts", "streamlines through the {ref} that pass the voxel"),
    ("tract_{cls}_{ref}", "count", "tracts", "of those, how many are {cls}"),
    ("align_{ref}", "0-1", "tracts", "|local fiber direction · direction away from the {ref}|; missing where no streamline passes"),
    ("density", "count", "tracts", "streamlines of the whole atlas passing the voxel"),
    ("n_bundles", "count", "tracts", "different bundles passing the voxel"),
    ("tract_end", "mm", "tracts", "distance to the nearest streamline end"),
    ("bundle_prob", "0-1", "tracts", "highest probability of belonging to any bundle (HCP-1065 maps)"),
    ("parcel_conn", "0-1", "connectivity", "share of the voxel's Yale parcel's connectivity that runs into parcels the whole tumor holds, each weighted by how much of it the tumor holds; missing outside the parcels"),
    ("parcel_tumor", "0-1", "connectivity", "share of the voxel's Yale parcel inside the whole tumor; missing outside the parcels"),
] + [(m.lower(), "a.u.", "structural intensity", f"pre-op {m}, raw 16-bit") for m in STRUCTURAL] + [
    ("flair_gradient", "a.u./mm", "structural intensity", "FLAIR gradient magnitude"),
    ("flair_sd", "a.u.", "structural intensity", "FLAIR standard deviation in the 3 mm box"),
    ("t1gd_sd", "a.u.", "structural intensity", "T1-Gd standard deviation in the 3 mm box"),
    ("fa", "0-1", "diffusion", "fractional anisotropy; missing if the scan has no diffusion"),
    ("md", "mm²/s", "diffusion", "mean diffusivity, the release's trace divided by 3"),
    ("ad", "mm²/s", "diffusion", "axial diffusivity"),
    ("rd", "mm²/s", "diffusion", "radial diffusivity"),
    ("rcbv", "a.u.", "perfusion", "relative cerebral blood volume (ap-rCBV); missing if the scan has no perfusion"),
    ("ph", "a.u.", "perfusion", "peak height"),
    ("psr", "a.u.", "perfusion", "percentage of signal recovery"),
    ("y_et", "0/1", "label", "enhancing tumor at follow-up"),
    ("y_tc", "0/1", "label", "tumor core at follow-up"),
    ("y_wt", "0/1", "label", "any tumor at follow-up"),
    ("y_new", "0/1", "label", "tumor at follow-up outside the pre-op whole tumor"),
]


def columns():
    out = []
    for name, units, group, what in SPEC:
        refs = REFERENCES if "{ref}" in name else ((None, None),)
        clss = CLASSES if "{cls}" in name else (None,)
        for ref, region in refs:
            for cls in clss:
                label = {"wt": "whole tumor", "core": "core"}.get(ref, "")
                out.append({"name": name.format(ref=ref, cls=cls), "units": units, "group": group,
                            "what": what.format(ref=label, cls=cls)})
    return out


def write_spec():
    FEATURES.mkdir(parents=True, exist_ok=True)
    spec = {"version": VERSION, "grid": "MNI152 1 mm, rows = template brain voxels",
            "inputs": {"segmentation": "BraTS 2025 winner (BraTS25_1), outputs/scans/*/seg_in_mni.nii.gz",
                       "registration": "each scan onto MNI: outputs/scans/*/affine.mat + warp.nii.gz (greedy)",
                       "intensities": "TCIA 16-bit NIfTI, images_structural / images_DTI / images_DSC",
                       "tracts": "HCP-1065, all 479,804 streamlines of the 77 non-cranial bundles [yeh22]",
                       "tissue": "SynthSeg, outputs/scans/*/tissue_in_mni.nii.gz", "lobes": "outputs/atlas/lobes_on_mni.nii.gz",
                       "parcels": "Yale Brain Atlas on MNI, structural connectivity SC_Count"},
            "missing": "NaN in float columns", "columns": columns()}
    (FEATURES / "spec.json").write_text(json.dumps(spec, indent=1))


# atlas-level values, once for every patient

def atlas_cache():
    if ATLAS_CACHE.exists():
        held = np.load(ATLAS_CACHE)
        return {k: held[k] for k in held.files}
    print("building the atlas cache", flush=True)
    mni = nib.load(MNI)
    shape = mni.shape
    data = hcp.everything()
    points, offsets, bundle, names = data["points"], data["offsets"], data["bundle"], data["names"]
    lines = len(offsets) - 1
    owner = np.repeat(np.arange(lines, dtype=np.int32), np.diff(offsets))
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(mni.affine), points)).astype(np.int32)
    ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
    flat = np.full(len(points), -1, np.int32)
    flat[ok] = np.ravel_multi_index(tuple(ijk[ok].T), shape)
    nvox = int(np.prod(shape))
    group = {p.name.removesuffix(".trk.gz"): p.parent.name for p in (ATLAS_INPUT / "hcp1065" / "trk").glob("*/*.trk.gz")}
    line_class = np.array([CLASSES.index(group[str(names[b])]) for b in bundle], np.int8)

    pairs = np.unique(owner[ok].astype(np.int64) * nvox + flat[ok])
    density = np.bincount(pairs % nvox, minlength=nvox).astype(np.float32)
    pairs = np.unique(bundle[owner[ok]].astype(np.int64) * nvox + flat[ok])
    n_bundles = np.bincount(pairs % nvox, minlength=nvox).astype(np.float32)
    del pairs

    ends = np.zeros(nvox, bool)
    for idx in (offsets[:-1], offsets[1:] - 1):
        good = flat[idx] >= 0
        ends[flat[idx][good]] = True
    tract_end = ndimage.distance_transform_edt(~ends.reshape(shape)).astype(np.float32).ravel()

    # local fiber direction: the main axis of the summed outer products of the
    # streamline directions passing each voxel, so opposite senses agree
    step = np.diff(points, axis=0)
    same = owner[1:] == owner[:-1]
    keep = same & ok[:-1]
    d = step[keep] / np.maximum(np.linalg.norm(step[keep], axis=1, keepdims=True), 1e-6)
    where = flat[:-1][keep]
    tensor = np.zeros((nvox, 6), np.float32)
    for k, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        tensor[:, k] = np.bincount(where, weights=d[:, a] * d[:, b], minlength=nvox)
    has = tensor[:, :3].sum(1) > 0
    T = tensor[has]
    full = np.stack([np.stack([T[:, 0], T[:, 3], T[:, 4]], -1), np.stack([T[:, 3], T[:, 1], T[:, 5]], -1),
                     np.stack([T[:, 4], T[:, 5], T[:, 2]], -1)], 1)
    direction = np.full((nvox, 3), np.nan, np.float32)
    direction[has] = np.linalg.eigh(full)[1][:, :, -1]
    del tensor, T, full, step, d, where

    # the bundle probability maps are on the ICBM template; take the highest, then move it
    maps = sorted((ATLAS_INPUT / "hcp1065" / "prob" / "prob").glob("*.nii.gz"))
    first = nib.load(maps[0])
    best = np.zeros(first.shape, np.float32)
    for m in maps:
        best = np.maximum(best, np.asarray(nib.load(m).dataobj, np.float32))
    work = FEATURES / "atlas_work"
    work.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(best, first.affine), work / "prob_icbm.nii.gz")
    subprocess.run(["greedy", "-d", "3", "-rf", str(MNI), "-ri", "LINEAR", "-rm", str(work / "prob_icbm.nii.gz"),
                    str(work / "prob_mni.nii.gz"), "-r", str(ATLAS_DIR / "icbm_to_mni_warp.nii.gz"),
                    str(ATLAS_DIR / "icbm_to_mni_affine.mat"), "-threads", "8"], capture_output=True, check=True)
    bundle_prob = np.asarray(nib.load(work / "prob_mni.nii.gz").dataobj, np.float32).ravel()
    shutil.rmtree(work)

    cache = {"flat": flat, "owner": owner, "start": offsets[:-1].astype(np.int64), "end": (offsets[1:] - 1).astype(np.int64),
             "line_class": line_class, "density": density, "n_bundles": n_bundles, "tract_end": tract_end,
             "direction": direction, "bundle_prob": bundle_prob}
    np.savez(ATLAS_CACHE, **cache)
    return cache


ATLAS = None


# per patient

def _to_mni(scan, work):
    # every TCIA image of this scan onto MNI in one greedy call, through the
    # scan's own registration; values linearly, the brain mask as a label
    moved, pairs, support = {}, [], {}
    sources = [(m.lower(), TCIA / "images_structural" / scan / f"{scan}_{m}.nii.gz") for m in STRUCTURAL]
    sources += [(m.lower(), TCIA / "images_DTI" / "images_DTI" / scan / f"{scan}_DTI_{m}.nii.gz") for m in DTI]
    sources += [(m.lower(), TCIA / "images_DSC" / scan / f"{scan}_DSC_{m}.nii.gz") for m in DSC]
    for name, src in sources:
        if src.exists():
            dest = work / f"{name}.nii.gz"
            pairs += ["-rm", str(src), str(dest)]
            moved[name] = dest
            if name not in [m.lower() for m in STRUCTURAL]:
                # the maps write 0 where they have no data; carry where they do,
                # so their edge can be told from real values
                image = nib.load(src)
                has = work / f"{name}_has.nii.gz"
                nib.save(nib.Nifti1Image((np.asarray(image.dataobj) != 0).astype(np.float32), image.affine), has)
                pairs += ["-rm", str(has), str(work / f"{name}_has_mni.nii.gz")]
                support[name] = work / f"{name}_has_mni.nii.gz"
    brain = work / "brain.nii.gz"
    args = ["greedy", "-d", "3", "-rf", str(MNI), "-ri", "LINEAR", *pairs,
            "-ri", "LABEL", "0.2vox", "-rm", str(out_dir(scan) / "brainmask.nii.gz"), str(brain),
            "-r", str(out_dir(scan) / "warp.nii.gz"), str(out_dir(scan) / "affine.mat"), "-threads", THREADS]
    subprocess.run(args, capture_output=True, check=True)
    out = {}
    for k, v in moved.items():
        values = np.asarray(nib.load(v).dataobj, np.float32)
        if k in support:
            # missing outside the map's own data, and where its edge blended in zeros
            values[np.asarray(nib.load(support[k]).dataobj) < 0.999] = np.nan
        out[k] = values
    return out, np.asarray(nib.load(brain).dataobj) > 0


def _edt(mask):
    return ndimage.distance_transform_edt(~mask).astype(np.float32)


def _signed(mask):
    if not mask.any():
        return np.full(mask.shape, np.nan, np.float32)
    return _edt(mask) - _edt(~mask)


def _unit_gradient(volume):
    g = np.stack(np.gradient(np.nan_to_num(volume)), -1)
    return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-6)


def _propagate(values_at, edge):
    # each voxel takes the value at its nearest edge voxel
    _, idx = ndimage.distance_transform_edt(~edge, return_indices=True)
    return values_at[tuple(idx)]


def _tracts(region_mask, shape):
    # streamlines through the region: per voxel, how many pass, by class, and the
    # shortest distance along any of them from the region
    a = ATLAS
    flat, owner = a["flat"], a["owner"]
    nvox = int(np.prod(shape))
    good = flat >= 0
    inside = np.zeros(len(flat), bool)
    inside[good] = region_mask.ravel()[flat[good]]
    crossing = np.bincount(owner[inside], minlength=len(a["start"])) > 0
    sel = np.flatnonzero(crossing[owner])
    out = {}
    v = flat[sel]
    gs = v >= 0
    pairs = np.unique(owner[sel][gs].astype(np.int64) * nvox + v[gs])
    vox, line = pairs % nvox, pairs // nvox
    out["n"] = np.bincount(vox, minlength=nvox).astype(np.float32)
    for c, cls in enumerate(CLASSES):
        m = a["line_class"][line] == c
        out[cls] = np.bincount(vox[m], minlength=nvox).astype(np.float32)
    # along-line distance: points are 1 mm apart within a line
    ins = inside[sel]
    start, end = a["start"][owner[sel]], a["end"][owner[sel]]
    last = np.maximum.accumulate(np.where(ins, sel, -1))
    nxt = np.minimum.accumulate(np.where(ins, sel, np.iinfo(np.int64).max)[::-1])[::-1]
    big = np.float32(np.inf)
    d = np.minimum(np.where(last >= start, (sel - last).astype(np.float32), big),
                   np.where(nxt <= end, (nxt - sel).astype(np.float32), big))
    dist = np.full(nvox, np.inf, np.float32)
    np.minimum.at(dist, v[gs], d[gs])
    dist[~np.isfinite(dist)] = np.nan
    out["dist"] = dist
    return out


def build(first):
    later = measure.later_of(first)
    n = first[10:15]
    here = FEATURES / first[:15]
    if (here / "voxels.npz").exists():
        return n, "already done"
    t0 = time.time()
    here.mkdir(parents=True, exist_ok=True)
    work = here / "work"
    work.mkdir(exist_ok=True)
    mni = nib.load(MNI)
    shape = mni.shape
    rows = template_brain()
    index = np.flatnonzero(rows.ravel()).astype(np.int32)
    cols = {}

    def put(name, volume, dtype=np.float32):
        cols[name] = np.asarray(volume).ravel()[index].astype(dtype)

    seg = np.asarray(nib.load(out_dir(first) / "seg_in_mni.nii.gz").dataobj).astype(np.int16)
    masks = {"WT": seg > 0, "TC": np.isin(seg, (1, 3)), "ET": seg == 3, "NETC": seg == 1}
    tissue = np.asarray(nib.load(out_dir(first) / "tissue_in_mni.nii.gz").dataobj).astype(np.int16)
    images, brain = _to_mni(first, work)

    depth = _edt(~brain)
    depth_dir = _unit_gradient(depth)
    # the template's axes are not rotated, so x depends on the first index only
    x = mni.affine[0, 0] * np.arange(shape[0]) + mni.affine[0, 3]
    xs = np.broadcast_to(x[:, None, None], shape)
    # the side holding most of the tumor; the midline plane counts for neither
    left = (masks["WT"] & (xs < 0)).sum() >= (masks["WT"] & (xs > 0)).sum()
    side = -1.0 if left else 1.0
    put("midline", xs * side)

    for ref, region in REFERENCES:
        m = masks[region]
        signed = _signed(m)
        put(f"dist_{ref}", signed)
        if m.any():
            edge = outline(m)
            pts = nib.affines.apply_affine(mni.affine, np.argwhere(edge))
            put(f"hull_{ref}", _hull(pts, mni, shape, rows) | m, np.uint8)
            away = _unit_gradient(signed)
            put(f"layer_{ref}", np.abs((away * depth_dir).sum(-1)))
            smooth = ndimage.gaussian_filter(np.nan_to_num(signed), 2.0)
            put(f"curvature_{ref}", _propagate(ndimage.laplace(smooth) / 2, edge))
            across = m & (xs * side < 0)
            put(f"across_{ref}", _edt(across) if across.any() else np.full(shape, np.nan))
        else:
            for c in ("hull", "layer", "curvature", "across"):
                put(f"{c}_{ref}", np.full(shape, np.nan))
            away = None
        tr = _tracts(m, shape) if m.any() else None
        put(f"tract_dist_{ref}", tr["dist"] if tr else np.full(shape, np.nan))
        put(f"tract_n_{ref}", tr["n"] if tr else np.zeros(shape))
        for cls in CLASSES:
            put(f"tract_{cls}_{ref}", tr[cls] if tr else np.zeros(shape))
        if away is not None:
            fiber = ATLAS["direction"].reshape(*shape, 3)
            put(f"align_{ref}", np.abs(np.einsum("...k,...k->...", np.nan_to_num(fiber), away)) /
                np.where(np.isnan(fiber[..., 0]), np.nan, 1.0))
        else:
            put(f"align_{ref}", np.full(shape, np.nan))
        del tr

    if masks["TC"].any():
        core_edge = outline(masks["TC"])
        put("halo_at_core", _propagate(_edt(~masks["WT"]), core_edge))
    else:
        put("halo_at_core", np.full(shape, np.nan))
    put("compartment", seg, np.uint8)
    put("dist_et", _signed(masks["ET"]))
    put("dist_netc", _signed(masks["NETC"]))

    put("depth", depth)
    put("in_brain", brain, np.uint8)
    put("dist_ventricles", _edt(np.isin(tissue, VENTRICLES)))
    put("dist_lateral", _edt(np.isin(tissue, LATERAL_VENTRICLE)))
    put("lobe", np.asarray(nib.load(ATLAS_DIR / "lobes_on_mni.nii.gz").dataobj), np.uint8)
    put("tissue", tissue, np.uint8)
    wm = np.isin(tissue, WM)
    border = wm & ndimage.binary_dilation(np.isin(tissue, CORTEX))
    gw = _edt(border)
    put("gw_border", np.where(wm, gw, -gw))
    put("sulcal_csf", _edt(np.isin(tissue, SULCAL_CSF)))
    put("wm_share", ndimage.uniform_filter(wm.astype(np.float32), WM_BOX_MM))

    for name in ("density", "n_bundles", "tract_end", "bundle_prob"):
        cols[name] = ATLAS[name][index].astype(np.float32)

    labels, names = label_volume(mni)
    held = np.bincount(labels.ravel(), weights=masks["WT"].ravel(), minlength=len(names) + 1)
    size = np.bincount(labels.ravel(), minlength=len(names) + 1)
    frac = np.divide(held, size, out=np.zeros_like(held), where=size > 0)[1:]
    W = matrix()
    total = W.sum(1)
    conn = np.divide(W @ frac, total, out=np.zeros(len(frac)), where=total > 0)
    lookup_conn = np.r_[np.nan, conn].astype(np.float32)
    lookup_frac = np.r_[np.nan, frac].astype(np.float32)
    put("parcel_conn", lookup_conn[labels])
    put("parcel_tumor", lookup_frac[labels])

    nan = np.full(shape, np.nan, np.float32)
    for m in STRUCTURAL:
        put(m.lower(), images.get(m.lower(), nan))
    flair, t1gd = images.get("flair"), images.get("t1gd")
    put("flair_gradient", np.linalg.norm(np.stack(np.gradient(flair), -1), axis=-1) if flair is not None else nan)
    sd = lambda v: np.sqrt(np.maximum(ndimage.uniform_filter(v * v, 3) - ndimage.uniform_filter(v, 3) ** 2, 0))
    put("flair_sd", sd(flair) if flair is not None else nan)
    put("t1gd_sd", sd(t1gd) if t1gd is not None else nan)
    put("fa", images.get("fa", nan))
    put("md", images["tr"] / 3 if "tr" in images else nan)
    put("ad", images.get("ad", nan))
    put("rd", images.get("rd", nan))
    put("rcbv", images.get("ap-rcbv", nan))
    put("ph", images.get("ph", nan))
    put("psr", images.get("psr", nan))

    after = np.asarray(nib.load(out_dir(later) / "seg_in_mni.nii.gz").dataobj).astype(np.int16)
    put("y_et", after == 3, np.uint8)
    put("y_tc", np.isin(after, (1, 3)), np.uint8)
    put("y_wt", after > 0, np.uint8)
    put("y_new", (after > 0) & ~masks["WT"], np.uint8)

    # patient-level values
    ml = ml_per_voxel(mni)
    normal = wm & brain & (_edt(masks["WT"]) > NEAR_MM)
    refs = {}
    for name, vol in images.items():
        v = vol[normal & np.isfinite(vol) & (vol > 0)]
        refs[name] = {"mean": float(v.mean()), "sd": float(v.std()), "median": float(np.median(v))} if len(v) else None
    clinical = {}
    with open(PARTICIPANTS) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["ID"] == first:
                clinical = {k: r[k] for k in ("Gender", "Age_at_scan_years", "IDH1", "MGMT", "KPS", "GTR_over90percent")}
    vent = _edt(np.isin(tissue, VENTRICLES))
    lobes = np.asarray(nib.load(ATLAS_DIR / "lobes_on_mni.nii.gz").dataobj)[masks["WT"]]
    patient = {
        "patient": first[:15], "pre_op": first, "follow_up": later, "version": VERSION,
        "mni_ml": {k: round(float(v.sum()) * ml, 2) for k, v in masks.items()} | {"SNFH": round(float((seg == 2).sum()) * ml, 2)},
        "across_midline_ml": {r: round(float((masks[r] & (xs * side < 0)).sum()) * ml, 2) for r in ("WT", "TC")},
        "main_side": "left" if left else "right",
        "tumor_depth_mm": round(float(np.median(depth[masks["WT"]])), 1),
        "lobe_mode": int(np.bincount(lobes[lobes > 0]).argmax()) if (lobes > 0).any() else None,
        "ventricle_contact_mm": round(float(vent[masks["WT"]].min()), 1),
        "normal_wm_reference": refs,
        "maps": sorted(images),
        "pathology_score": measure.pathology(first),
        "clinical_optional": clinical,
    }
    (here / "patient.json").write_text(json.dumps(patient, indent=1))

    # a few columns as images, to look at
    qc = here / "qc"
    qc.mkdir(exist_ok=True)
    for name in ("dist_wt", "tract_dist_wt", "tract_n_wt", "layer_wt", "curvature_wt", "depth", "align_wt", "fa", "rcbv", "flair"):
        volume = np.full(int(np.prod(shape)), np.nan, np.float32)
        volume[index] = cols[name]
        nib.save(nib.Nifti1Image(volume.reshape(shape), mni.affine), qc / f"{name}.nii.gz")

    tmp = here / "voxels.tmp.npz"
    np.savez_compressed(tmp, index=index, **cols)
    tmp.replace(here / "voxels.npz")
    shutil.rmtree(work)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return n, f"done in {time.time() - t0:.0f} s, peak {peak:.0f} MB, {len(cols)} columns"


def _on_mains():
    for p in Path("/sys/class/power_supply").glob("AC*/online"):
        return p.read_text().strip() == "1"
    for p in Path("/sys/class/power_supply").glob("*/online"):
        if p.read_text().strip() == "1":
            return True
    return True


def _free_gb():
    with open("/proc/meminfo") as fh:
        info = dict(line.split(":", 1) for line in fh)
    return int(info["MemAvailable"].split()[0]) / 2 ** 20


def _worker(first):
    # never start a patient on battery or without room
    while not _on_mains() or _free_gb() < 3:
        time.sleep(30)
    try:
        return build(first)
    except Exception as error:
        return first[10:15], f"FAILED: {error!r}"


def main():
    global ATLAS
    parser = argparse.ArgumentParser()
    parser.add_argument("patients", nargs="*")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    write_spec()
    ATLAS = atlas_cache()
    firsts = [f for f, _ in measure.pairs("native")]
    if args.patients:
        firsts = [f for f in firsts if f[10:15] in args.patients]
    if args.workers == 1:
        for f in firsts:
            print(*_worker(f), flush=True)
        return
    # fork, so the workers share the atlas cache instead of each loading it
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for n, status in pool.imap_unordered(_worker, firsts):
            print(n, status, flush=True)


if __name__ == "__main__":
    main()
