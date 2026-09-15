# Yale Brain Atlas: official parcel volume, meshes, metadata and overlap.

import csv
import urllib.request

import nibabel as nib
import numpy as np

from paths import ATLAS_DIR, ATLAS_INPUT, LABELS

BASE = "https://raw.githubusercontent.com/YaleBrainAtlas/YaleBrainAtlas/master/data"

FILES = {
    "YBA_696.nii": f"{BASE}/YBA_696parcels/YBA_696.nii",
    "YBA_696_LUT.txt": f"{BASE}/YBA_696parcels/YBA_696_LUT.txt",
    "YBA_696_parcel_dict.csv": f"{BASE}/YBA_696parcels/YBA_696_parcel_dict.csv",
    "YBA_696.glb": f"{BASE}/YBA_696parcels/YBA_696.glb",
    "structuralconnectivity_metrics_mean.csv":
        f"{BASE}/Connectome/structuralconnectivity_metrics_mean.csv",
}

RESAMPLED = ATLAS_DIR / "yba_on_mni.nii.gz"

# glTF stores the atlas with x and z negated relative to the parcel dictionary
GLB_FLIP = np.array([-1.0, 1.0, -1.0])


def ensure(name):
    # Fetch one official file into inputs/atlas if it is not already there.
    path = ATLAS_INPUT / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"    downloading {name} ...", end="", flush=True)
        request = urllib.request.Request(FILES[name],
                                         headers={"User-Agent": "brats-upenn-gbm"})
        with urllib.request.urlopen(request, timeout=600) as response, open(path, "wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
        print(" done")
    return path


def parcels():
    # Official parcel metadata, in label order: label value i is parcels()[i-1].
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


def meshes():
    # Per-parcel surfaces from the official GLB, in label order.
    import trimesh
    scene = trimesh.load(ensure("YBA_696.glb"))
    out = []
    for key in list(scene.geometry):
        mesh = scene.geometry[key].copy()
        mesh.vertices = np.asarray(mesh.vertices) * GLB_FLIP
        out.append(mesh)
    return out


def label_volume(reference):
    # Official parcel volume, nearest-neighbour onto the reference grid.
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
    # Per-parcel tumour volume in mL, most involved first.
    meta = {p["name"]: p for p in parcels()}
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
        info = meta[name]
        rows.append({
            "parcel": name,
            "long_name": info["long_name"],
            "lobe": info["lobe"],
            "gyrus": info["gyrus"],
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
