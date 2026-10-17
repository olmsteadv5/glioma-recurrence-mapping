
import csv
import functools
import urllib.request

import nibabel as nib
import numpy as np

from geometry import ml_per_voxel
from paths import ATLAS_DIR, ATLAS_INPUT, LABELS, MNI

BASE = "https://raw.githubusercontent.com/YaleBrainAtlas/YaleBrainAtlas/master/data"

FILES = {
    "YBA_696.nii": f"{BASE}/YBA_696parcels/YBA_696.nii",
    "YBA_696_parcel_dict.csv": f"{BASE}/YBA_696parcels/YBA_696_parcel_dict.csv",
}

RESAMPLED = ATLAS_DIR / "yba_on_mni.nii.gz"


def ensure(name):
    path = ATLAS_INPUT / name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name}")
    request = urllib.request.Request(FILES[name],
                                     headers={"User-Agent": "brats-upenn-gbm"})
    with urllib.request.urlopen(request, timeout=600) as response, open(path, "wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    return path


@functools.lru_cache(maxsize=4)
def parcels():
    # explainer: label i is parcels()[i-1]
    with open(ensure("YBA_696_parcel_dict.csv")) as fh:
        rows = list(csv.DictReader(fh))
    return [{
        "name": r["Name"],
        "long_name": r["Long_name"],
        "hemisphere": r["Hemisphere"],
        "lobe": r["Lobe"],
        "region": r["Region"],
        "gyrus": r["Gyrus"],
        "color": (int(r["YBA_R_color"]), int(r["YBA_G_color"]), int(r["YBA_B_color"])),
    } for r in rows]


def brain():
    # decision: one closed surface, stacking the parcel meshes piles up alpha until nothing shows through
    cache = ATLAS_DIR / "brain_surface.npz"
    if cache.exists():
        held = np.load(cache)
        return held["vertices"], held["faces"]

    import pyvista as pv
    image = nib.load(MNI)
    mask = (np.asarray(image.dataobj) > 0).astype(np.float32)
    grid = pv.ImageData(dimensions=np.array(mask.shape) + 1)
    grid.cell_data["v"] = mask.flatten(order="F")
    surface = grid.cells_to_points("v").contour([0.5], scalars="v")
    surface = surface.smooth(n_iter=40, relaxation_factor=0.2).decimate(0.8)

    vertices = nib.affines.apply_affine(image.affine, surface.points)
    faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, vertices=vertices, faces=faces)
    return vertices, faces


def label_volume(reference):
    names = [p["name"] for p in parcels()]
    if RESAMPLED.exists():
        return np.asarray(nib.load(RESAMPLED).dataobj).astype(np.int32), names

    atlas_img = nib.load(ensure("YBA_696.nii"))
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
    meta = {p["name"]: p for p in parcels()}
    image = nib.load(seg_mni)
    tumor = np.asarray(image.dataobj).astype(np.int16)
    voxel_ml = ml_per_voxel(image)
    hit = tumor > 0

    total = np.bincount(labels.ravel(), minlength=len(names) + 1)
    inside = np.bincount(labels[hit].ravel(), minlength=len(names) + 1)

    rows = []
    for index, name in enumerate(names, start=1):
        if inside[index] == 0:
            continue
        tissue = tumor[hit & (labels == index)]
        counts = {LABELS.get(int(v), str(v)): int((tissue == v).sum())
                  for v in np.unique(tissue)}
        info = meta[name]
        rows.append({
            "parcel": name,
            "long_name": info["long_name"],
            "lobe": info["lobe"],
            "gyrus": info["gyrus"],
            "parcel_ml": round(total[index] * voxel_ml, 3),
            "in_tumor_ml": round(inside[index] * voxel_ml, 3),
            "fraction": round(float(inside[index]) / total[index], 4),
            "tissue": max(counts, key=counts.get),
        })

    rows.sort(key=lambda r: (-r["fraction"], -r["in_tumor_ml"]))
    return {
        "tumor_ml": round(float(hit.sum()) * voxel_ml, 2),
        # explainer: gray matter only, so well under the tumor volume
        "in_parcels_ml": round(sum(r["in_tumor_ml"] for r in rows), 2),
        "parcels": rows,
    }
