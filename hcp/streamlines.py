# explainer: HCP-1065 streamlines, made in ICBM152 so they get registered onto our MNI brain first

import functools
import subprocess

import nibabel as nib
import numpy as np

from paths import ATLAS_DIR, ATLAS_INPUT, MNI, THREADS

SOURCE = ATLAS_INPUT / "hcp1065"
CACHE = ATLAS_DIR / "hcp1065_streamlines.npz"
EVERY = ATLAS_DIR / "hcp1065_all_streamlines.npz"
TEMPLATE = ATLAS_DIR / "icbm_brain.nii.gz"
AFFINE = ATLAS_DIR / "icbm_to_mni_affine.mat"
WARP = ATLAS_DIR / "icbm_to_mni_warp.nii.gz"
WARP_INVERSE = ATLAS_DIR / "icbm_to_mni_warp_inverse.nii.gz"

# decision: skip cranial nerves, they run outside the brain where no plan grows
GROUPS = ("association", "commissural", "projection", "cerebellum")
# decision: fixed 1 in 10 of every bundle, same ones every run. checked against the full atlas, comparisons hold
KEEP = 10
SEED = 0
# decision: one step per voxel so a line can't skip a voxel
STEP_MM = 1.0


def _need(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Download hcp1065_avg_tracts_trk.zip (unzip into "
            f"trk/), ICBM152_adult.T1W.nii.gz and ICBM152_adult.mask.nii.gz from "
            f"github.com/data-others/atlas/releases into {SOURCE}; see the README")
    return path


def template():
    if TEMPLATE.exists():
        return TEMPLATE
    from nibabel.processing import resample_from_to

    t1 = nib.load(_need(SOURCE / "ICBM152_adult.T1W.nii.gz"))
    mask = resample_from_to(nib.load(_need(SOURCE / "ICBM152_adult.mask.nii.gz")), t1, order=0)
    brain = np.asarray(t1.dataobj) * (np.asarray(mask.dataobj) > 0)
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(brain.astype(np.float32), t1.affine), TEMPLATE)
    return TEMPLATE


def register():
    # decision: same affine + deformable steps every scan takes onto MNI
    if WARP_INVERSE.exists():
        return
    common = ["-d", "3", "-i", str(MNI), str(template()), "-n", "100x50x20", "-threads", THREADS]
    subprocess.run(["greedy", *common, "-a", "-dof", "12", "-m", "NCC", "4x4x4",
                    "-ia-image-centers", "-o", str(AFFINE)], capture_output=True, check=True)
    subprocess.run(["greedy", *common, "-m", "NCC", "2x2x2", "-it", str(AFFINE),
                    "-o", str(WARP), "-oinv", str(WARP_INVERSE)], capture_output=True, check=True)


def _resample(line, step=STEP_MM):
    walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    if walked[-1] < step:
        return line
    want = np.arange(0, walked[-1], step)
    return np.stack([np.interp(want, walked, line[:, k]) for k in range(3)], axis=1)


def _to_mni(points, chunk=5_000_000):
    # explainer: greedy moves points the opposite way to images, so points need the inverse chain
    work = ATLAS_DIR / "hcp1065_points"
    work.mkdir(parents=True, exist_ok=True)
    source, dest = work / "icbm.vtk", work / "mni.vtk"
    moved = []
    for start in range(0, len(points), chunk):
        part = points[start:start + chunk]
        with open(source, "wb") as fh:
            fh.write(f"# vtk DataFile Version 3.0\nstreamlines\nBINARY\nDATASET POLYDATA\n"
                     f"POINTS {len(part)} float\n".encode())
            fh.write(part.astype(">f4").tobytes())
            fh.write(b"\n")
        subprocess.run(["greedy", "-d", "3", "-rf", str(template()), "-rs", str(source), str(dest),
                        "-r", f"{AFFINE},-1", str(WARP_INVERSE), "-threads", THREADS],
                       capture_output=True, check=True)
        text = dest.read_text().split("POINTS", 1)[1].split("\n", 1)[1]
        moved.append(np.array(text.split()[:3 * len(part)], dtype=np.float32).reshape(-1, 3))
    source.unlink()
    dest.unlink()
    work.rmdir()
    return np.vstack(moved)


def everything():
    # decision: load into memory once, indexing the npz rereads from disk every time
    if EVERY.exists():
        with np.load(EVERY) as held:
            return {k: held[k] for k in held.files}
    register()
    lines, bundle, names = [], [], []
    for group in GROUPS:
        for path in sorted((SOURCE / "trk" / group).glob("*.trk.gz")):
            every = nib.streamlines.load(path).streamlines
            lines += [_resample(np.asarray(line, float)).astype(np.float32) for line in every]
            bundle += [len(names)] * len(every)
            names.append(path.name.removesuffix(".trk.gz"))
    offsets = np.cumsum([0] + [len(l) for l in lines])
    points = _to_mni(np.vstack(lines))
    np.savez(EVERY, points=points, offsets=offsets,
             bundle=np.array(bundle, np.int16), names=np.array(names))
    return everything()


def pick(bundle, share, seed=SEED):
    # explainer: each bundle shuffled once per seed and taken from the front, so a smaller share is always inside a bigger one
    rng = np.random.default_rng(seed)
    out = []
    for b in range(int(bundle.max()) + 1):
        members = np.flatnonzero(bundle == b)
        order = rng.permutation(len(members))
        out.append(np.sort(members[order[:max(int(round(len(members) * share)), 1)]]))
    return np.concatenate(out)


def _load():
    if CACHE.exists():
        return np.load(CACHE)
    data = everything()
    points, offsets = data["points"], data["offsets"]
    keep = pick(data["bundle"], 1 / KEEP)
    lines = [points[offsets[i]:offsets[i + 1]] for i in keep]
    np.savez_compressed(CACHE, points=np.vstack(lines),
                        offsets=np.cumsum([0] + [len(l) for l in lines]),
                        bundle=data["bundle"][keep], names=data["names"])
    return np.load(CACHE)


@functools.lru_cache(maxsize=1)
def streamlines():
    # explainer: shared, don't write into these
    data = _load()
    return np.split(data["points"].astype(float), data["offsets"][1:-1])



def crossing(lines, seg_img, labels, involved):
    # explainer: lines through the mass, plus lines that miss it but clip an involved parcel
    tumor = np.asarray(seg_img.dataobj) > 0
    shape = np.array(tumor.shape)
    inverse = np.linalg.inv(seg_img.affine)
    wanted = list(set(involved))

    mass, nearby = [], []
    for line in lines:
        ijk = np.rint(nib.affines.apply_affine(inverse, line)).astype(int)
        inside = np.all((ijk >= 0) & (ijk < shape), axis=1)
        if not inside.any():
            continue
        sel = ijk[inside]
        if tumor[sel[:, 0], sel[:, 1], sel[:, 2]].any():
            mass.append(line)
        elif wanted and np.isin(labels[sel[:, 0], sel[:, 1], sel[:, 2]], wanted).any():
            nearby.append(line)
    return mass, nearby
