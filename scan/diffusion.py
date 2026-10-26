#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: tractography from each UPenn patient's own pre-op diffusion scan, put on the pre-op grid with no MNI step

import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hcp.streamlines import _resample
from scan.register import _greedy, _metric_mask
from paths import DCM2NIIX, DSI_STUDIO, INPUTS, PAIRS, THREADS, modality, out_dir, stripped_dir, subject_id

DTI = INPUTS / "upenn_dti"
SERIES = DTI / "series.json"
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID="
# decision: same streamline count as the atlas walk-down uses, so the two compare like for like
TRACTS = 48000


def raw(subject):
    return DTI / subject


def work(subject):
    path = out_dir(subject) / "diffusion"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dsi(args, log):
    with open(log, "a") as fh:
        fh.write(f"\n$ dsi_studio {' '.join(args)}\n")
        fh.flush()
        done = subprocess.run([str(DSI_STUDIO), *args], stdout=fh, stderr=subprocess.STDOUT,
                              env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    if done.returncode != 0:
        raise RuntimeError(f"dsi_studio {args[0]} failed, see {log}")


def pick(subject):
    # decision: the pre-op study is the earliest one. _T / _TERA / ONLY names are re-sends, so the plain series wins, then the most volumes
    pid = subject[:15]
    mine = [s for s in json.loads(SERIES.read_text()) if s["PatientID"] == pid]
    first = min(s["StudyDate"][:10] for s in mine)
    found = [s for s in mine if s["StudyDate"][:10] == first
             and re.search(r"DTI|DIFF|MDDW", s.get("SeriesDescription") or "", re.I)]
    found.sort(key=lambda s: (bool(re.search(r"_T\b|_TERA|ONLY", s["SeriesDescription"], re.I)), -s["ImageCount"]))
    return found[0] if found else None


def fetch(subject):
    d = raw(subject)
    if (d / "dwi.fz").exists():
        return True
    series = pick(subject)
    if series is None:
        return False
    d.mkdir(parents=True, exist_ok=True)
    log = d / "fetch.log"
    if not (d / "dwi.nii.gz").exists():
        archive = d / "series.zip"
        for _ in range(4):
            if (subprocess.run(["curl", "-sf", "--max-time", "900", "-o", str(archive), API + series["SeriesInstanceUID"]]).returncode == 0
                    and zipfile.is_zipfile(archive)):
                break
            time.sleep(10)
        else:
            raise RuntimeError(f"{subject}: download failed")
        zipfile.ZipFile(archive).extractall(d / "dicom")
        archive.unlink()
        subprocess.run([str(DCM2NIIX), "-z", "y", "-f", "dwi", "-o", str(d), str(d / "dicom")],
                       stdout=open(log, "a"), stderr=subprocess.STDOUT, check=True)
    _dsi(["--action=src", f"--source={d}/dwi.nii.gz", f"--bval={d}/dwi.bval", f"--bvec={d}/dwi.bvec", f"--output={d}/dwi.sz"], log)
    # explainer: qc writes qc.tsv into the working directory unless told otherwise
    _dsi(["--action=qc", f"--source={d}/dwi.sz", f"--output={d}/qc.tsv"], log)
    # decision: check_btable picks the gradient axis order that gives the most coherent fibers, one older protocol needs it
    _dsi(["--action=rec", f"--source={d}/dwi.sz", "--method=4", "--param=1.25", "--check_btable=1",
          f"--thread_count={THREADS}", f"--output={d}/dwi.fz"], log)
    return True


def dsi_to_world(subject):
    # explainer: dsi resamples to its own 2 mm grid and header, so map its voxels onto the scanner grid, flipping any axis that disagrees
    d = raw(subject)
    scan, grid = nib.load(d / "dwi.nii.gz"), nib.load(d / "dwi.fz.iso.nii.gz")
    zoom, size = np.array(scan.header.get_zooms()[:3]), np.array(scan.shape[:3])
    step = np.abs(np.diag(grid.affine)[:3])
    flip = np.sign(np.diag(grid.affine)[:3]) != np.sign(np.diag(scan.affine)[:3])
    to_scan = np.eye(4)
    for axis in range(3):
        to_scan[axis, axis] = (-1 if flip[axis] else 1) * step[axis] / zoom[axis]
        to_scan[axis, 3] = size[axis] - 1 if flip[axis] else 0
    corrected = scan.affine @ to_scan
    return corrected @ np.linalg.inv(grid.affine), corrected


def _silhouette(path, dest):
    image = nib.load(path)
    brain = ndimage.binary_fill_holes(ndimage.binary_closing(np.asarray(image.dataobj) > 0, iterations=2))
    nib.save(nib.Nifti1Image(ndimage.gaussian_filter(brain.astype(np.float32), 2 / np.array(image.header.get_zooms()[:3])), image.affine), dest)


def _rotation(matrix):
    m = np.loadtxt(matrix)[:3, :3]
    m = m / np.cbrt(abs(np.linalg.det(m)))
    return np.degrees(np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1)))


def _fa_contrast(subject, matrix, dest):
    # explainer: fa belongs in white matter, so a good fit puts high fa on SynthSeg's white matter and low fa on cortex
    w = work(subject)
    _greedy(["-d", 3, "-rf", modality(stripped_dir(subject), subject, "T2"), "-rm", w / "fa.nii.gz", dest, "-r", matrix,
             "-threads", THREADS], w / "register.log")
    fa = np.asarray(nib.load(dest).dataobj)
    tissue = np.asarray(nib.load(out_dir(subject) / "tissue.nii.gz").dataobj)
    return float(fa[np.isin(tissue, (2, 41))].mean() - fa[np.isin(tissue, (3, 42))].mean())


def register(subject, tries=4):
    # decision: dsi's iso map looks most like a T2, so it goes onto the same session's T2. rigid then affine, the affine soaks up some EPI distortion
    w = work(subject)
    if (w / "to_t2.mat").exists():
        return w / "to_t2.mat"
    d = raw(subject)
    _dsi(["--action=exp", f"--source={d}/dwi.fz", "--export=iso,fa"], w / "register.log")
    _, corrected = dsi_to_world(subject)
    for name in ("iso", "fa"):
        image = nib.load(d / f"dwi.fz.{name}.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(image.dataobj).astype(np.float32), corrected), w / f"{name}.nii.gz")
    t2 = modality(stripped_dir(subject), subject, "T2")
    _metric_mask(t2, out_dir(subject) / "seg.nii.gz", w / "mask.nii.gz")
    # decision: start from the two brain silhouettes, that fit is smooth and lands the same every run
    _silhouette(t2, w / "t2_shape.nii.gz")
    _silhouette(w / "iso.nii.gz", w / "iso_shape.nii.gz")
    _greedy(["-d", 3, "-a", "-dof", 6, "-m", "SSD", "-i", w / "t2_shape.nii.gz", w / "iso_shape.nii.gz",
             "-ia-image-centers", "-o", w / "shape.mat", "-n", "100x100x50", "-threads", THREADS], w / "register.log")
    # decision: NMI refine only at the two finest levels (coarse ones jumped up to 90 degrees), a few runs each, keep the fit with fa most on white matter
    start, best = w / "shape.mat", None
    for dof in (6, 12):
        scored = [(_fa_contrast(subject, start, w / "fa_try.nii.gz"), start)]
        for k in range(tries):
            out = w / f"try_{dof}_{k}.mat"
            _greedy(["-d", 3, "-a", "-dof", dof, "-m", "NMI", "-i", t2, w / "iso.nii.gz", "-gm", w / "mask.nii.gz",
                     "-ia", start, "-o", out, "-n", "20x50", "-threads", THREADS], w / "register.log")
            if _rotation(out) <= 30:
                scored.append((_fa_contrast(subject, out, w / "fa_try.nii.gz"), out))
        best = max(scored)
        start = best[1]
    if best[0] <= 0:
        raise RuntimeError(f"{subject}: no fit puts fa on white matter, see {w / 'register.log'}")
    shutil.copyfile(best[1], w / "to_t2.mat")
    for leftover in list(w.glob("try_*.mat")) + [w / "fa_try.nii.gz"]:
        leftover.unlink(missing_ok=True)
    _fa_contrast(subject, w / "to_t2.mat", w / "fa_on_t2.nii.gz")
    return w / "to_t2.mat"


def track(subject):
    w = work(subject)
    if not (w / "tracts.trk.gz").exists():
        # decision: whole brain, not tumor seeded, since fa drops in the edema and seeds there stop early
        _dsi(["--action=trk", f"--source={raw(subject)}/dwi.fz", f"--tract_count={TRACTS}", "--turning_angle=60",
              "--min_length=30", "--max_length=300", "--random_seed=1", f"--thread_count={THREADS}",
              f"--output={w}/tracts.trk.gz"], w / "track.log")
    return w / "tracts.trk.gz"


def to_native(subject):
    # explainer: trk points sit in dsi's frame. dsi -> scanner -> T2 grid, the last step through greedy's inverse since points move opposite to images
    w = work(subject)
    lines = [_resample(np.asarray(line, float)) for line in nib.streamlines.load(track(subject)).streamlines]
    offsets = np.cumsum([0] + [len(line) for line in lines])
    to_world, _ = dsi_to_world(subject)
    points = nib.affines.apply_affine(to_world, np.vstack(lines)).astype(np.float32)
    source, dest = w / "points.vtk", w / "points_on_t2.vtk"
    with open(source, "wb") as fh:
        fh.write(f"# vtk DataFile Version 3.0\nstreamlines\nBINARY\nDATASET POLYDATA\nPOINTS {len(points)} float\n".encode())
        fh.write(points.astype(">f4").tobytes())
        fh.write(b"\n")
    _greedy(["-d", 3, "-rf", modality(stripped_dir(subject), subject, "T2"), "-rs", source, dest,
             "-r", f"{w / 'to_t2.mat'},-1", "-threads", THREADS], w / "register.log")
    text = dest.read_text().split("POINTS", 1)[1].split("\n", 1)[1]
    moved = np.array(text.split()[:3 * len(points)], dtype=np.float32).reshape(-1, 3)
    source.unlink()
    dest.unlink()
    np.savez(w / "tracts_native.npz", points=moved, offsets=offsets)
    return moved


def check(subject, points):
    # decision: tracts belong in white matter, so the share of points landing there is the registration check
    t2 = nib.load(modality(stripped_dir(subject), subject, "T2"))
    tissue = np.asarray(nib.load(out_dir(subject) / "tissue.nii.gz").dataobj)
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(t2.affine), points)).astype(int)
    inside = np.all((ijk >= 0) & (ijk < t2.shape), axis=1)
    label = tissue[tuple(ijk[inside].T)]
    fa = np.asarray(nib.load(work(subject) / "fa_on_t2.nii.gz").dataobj)
    result = {"white_matter": round(float(np.isin(label, (2, 41)).mean()), 3),
              "cortex": round(float(np.isin(label, (3, 42)).mean()), 3),
              "outside_brain": round(float((label == 0).mean() + (1 - inside.mean())), 3),
              "fa_white_matter": round(float(fa[np.isin(tissue, (2, 41))].mean()), 3),
              "fa_cortex": round(float(fa[np.isin(tissue, (3, 42))].mean()), 3)}
    (work(subject) / "check.json").write_text(json.dumps(result, indent=1))
    return result


def run(number):
    subject = subject_id(number, 1)
    if not fetch(subject):
        return subject, "no pre-op diffusion"
    register(subject)
    return subject, check(subject, to_native(subject))


if __name__ == "__main__":
    numbers = [int(a) for a in sys.argv[1:]] or [int(p) for p in sorted(os.listdir(PAIRS)) if p.isdigit() and int(p) < 10000]
    for n in numbers:
        try:
            print(*run(n), flush=True)
        except Exception as error:
            print(subject_id(n, 1), "failed:", error, flush=True)
