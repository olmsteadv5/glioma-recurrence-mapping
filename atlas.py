"""Volume intersection between a tumour and the Yale Brain Atlas parcels."""

import json

import nibabel as nib
import numpy as np

from paths import ATLAS_DIR, LABELS

LABEL_VOLUME = ATLAS_DIR / "yba_labels_mni.nii.gz"
LABEL_NAMES = ATLAS_DIR / "yba_labels_mni.names.json"


def load_atlas():
    """The atlas and its plotly visualiser."""
    from YaleBrainAtlas import YaleBrainAtlas, YBAVisualizer
    yba = YaleBrainAtlas("Subject")
    return yba, YBAVisualizer(yba)


def label_volume(yba, reference):
    """Voxelise every parcel onto the MNI grid; cached, identical for all subjects."""
    if LABEL_VOLUME.exists() and LABEL_NAMES.exists():
        return (np.asarray(nib.load(LABEL_VOLUME).dataobj).astype(np.int32),
                json.loads(LABEL_NAMES.read_text()))

    print("    voxelising 696 parcels (one-off) ...", end="", flush=True)
    shape = reference.shape
    labels = np.zeros(shape, dtype=np.int32)
    names = sorted(yba.parcel_names)
    inverse = np.linalg.inv(reference.affine)

    for index, name in enumerate(names, start=1):
        mesh = getattr(yba, name).mesh
        corners = np.array([[x, y, z] for x in mesh.bounds[:, 0]
                            for y in mesh.bounds[:, 1] for z in mesh.bounds[:, 2]])
        voxels = nib.affines.apply_affine(inverse, corners)
        low = np.maximum(np.floor(voxels.min(axis=0)).astype(int) - 1, 0)
        high = np.minimum(np.ceil(voxels.max(axis=0)).astype(int) + 2, shape)
        if np.any(high <= low):
            continue

        grid = np.stack(np.meshgrid(*[np.arange(low[i], high[i]) for i in range(3)],
                                    indexing="ij"), axis=-1).reshape(-1, 3)
        inside = mesh.contains(nib.affines.apply_affine(reference.affine, grid))
        if not inside.any():
            continue
        selected = grid[inside]
        free = labels[selected[:, 0], selected[:, 1], selected[:, 2]] == 0
        selected = selected[free]
        labels[selected[:, 0], selected[:, 1], selected[:, 2]] = index

    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(labels.astype(np.int16), reference.affine), LABEL_VOLUME)
    LABEL_NAMES.write_text(json.dumps(names))
    print(f" done ({(labels > 0).sum() / 1000:.0f} mL parcelled)")
    return labels, names


def overlap(seg_mni, labels, names):
    """Per-parcel tumour volume in mL, most involved first."""
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
        # parcels cover only ~half the brain, so this is well under 100%
        "in_parcels_ml": round(sum(r["in_tumour_ml"] for r in rows), 2),
        "parcels": rows,
    }
