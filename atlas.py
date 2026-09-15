# Volume intersection between a tumour and the Yale Brain Atlas parcels.

import json
import re
import urllib.request

import nibabel as nib
import numpy as np

from paths import ATLAS_DIR, ATLAS_INPUT, LABELS

BASE = ("https://raw.githubusercontent.com/YaleBrainAtlas/YaleBrainAtlas/"
        "master/data/YBA_696parcels")
ATLAS_NII = ATLAS_INPUT / "YBA_696.nii"
ATLAS_LUT = ATLAS_INPUT / "YBA_696_LUT.txt"

RESAMPLED = ATLAS_DIR / "yba_on_mni.nii.gz"


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "brats-upenn-gbm"})
    with urllib.request.urlopen(request, timeout=300) as response, open(dest, "wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)


def load_atlas():
    # The atlas and its plotly visualiser, both from Ryan's wrapper package.
    from YaleBrainAtlas import YaleBrainAtlas, YBAVisualizer
    yba = YaleBrainAtlas("Subject")
    return yba, YBAVisualizer(yba)


def parcel_names():
    # Official label table; index i names label value i. Same order as yba.parcel_names.
    if not ATLAS_LUT.exists():
        _download(f"{BASE}/YBA_696_LUT.txt", ATLAS_LUT)
    table = {}
    for line in ATLAS_LUT.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = re.split(r"\s+", line.strip())
        table[int(parts[0])] = parts[1]
    return [table[i] for i in sorted(table) if i > 0]


def label_volume(reference):
    # Official 696-parcel volume, nearest-neighbour onto the reference grid.
    names = parcel_names()
    if RESAMPLED.exists():
        return np.asarray(nib.load(RESAMPLED).dataobj).astype(np.int32), names

    if not ATLAS_NII.exists():
        print("    downloading the official YBA parcel volume ...", end="", flush=True)
        _download(f"{BASE}/YBA_696.nii", ATLAS_NII)
        print(" done")

    atlas_img = nib.load(ATLAS_NII)
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32)

    grid = np.indices(reference.shape).reshape(3, -1).T
    world = nib.affines.apply_affine(reference.affine, grid)
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(atlas_img.affine), world)).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(atlas_img.shape)), axis=1)

    labels = np.zeros(len(grid), dtype=np.int32)
    picked = ijk[inside]
    labels[inside] = atlas_data[picked[:, 0], picked[:, 1], picked[:, 2]]
    labels = labels.reshape(reference.shape)

    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(labels.astype(np.int16), reference.affine), RESAMPLED)
    return labels, names


def overlap(seg_mni, labels, names):
    # Per-parcel tumour volume in mL, most involved first.
    image = nib.load(seg_mni)
    tumour = np.asarray(image.dataobj).astype(np.int16)
    voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0
    hit = tumour > 0

    total = np.bincount(labels.ravel(), minlength=len(names) + 1)
    inside = np.bincount(labels[hit].ravel(), minlength=len(names) + 1)

    rows = []
    for index, name in enumerate(names, start=1):
        if inside[index] == 0:
            continue
        tissue = tumour[hit & (labels == index)]
        counts = {LABELS.get(int(v), str(v)): int((tissue == v).sum())
                  for v in np.unique(tissue)}
        rows.append({
            "parcel": name,
            "parcel_ml": round(total[index] * voxel_ml, 3),
            "in_tumour_ml": round(inside[index] * voxel_ml, 3),
            "fraction": round(float(inside[index]) / total[index], 4),
            "tissue": max(counts, key=counts.get),
        })

    rows.sort(key=lambda r: (-r["fraction"], -r["in_tumour_ml"]))
    return {
        "tumour_ml": round(float(hit.sum()) * voxel_ml, 2),
        # parcels cover grey matter only, so this is well under 100%
        "in_parcels_ml": round(sum(r["in_tumour_ml"] for r in rows), 2),
        "parcels": rows,
    }
